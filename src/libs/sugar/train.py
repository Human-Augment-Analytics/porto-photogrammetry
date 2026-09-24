import os
import argparse
import json
from sugar_utils.general_utils import str2bool
from sugar_trainers.coarse_density import coarse_training_with_density_regularization
from sugar_trainers.coarse_sdf import coarse_training_with_sdf_regularization
from sugar_trainers.coarse_density_and_dn_consistency import coarse_training_with_density_regularization_and_dn_consistency
from sugar_extractors.coarse_mesh import extract_mesh_from_coarse_sugar
from sugar_trainers.refine import refined_training
from sugar_extractors.refined_mesh import extract_mesh_and_texture_from_refined_sugar


class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self


if __name__ == "__main__":
    # ----- Parser -----
    parser = argparse.ArgumentParser(description='Script to optimize a full SuGaR model.')
    
    # Data and vanilla 3DGS checkpoint
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='(Required) path to the scene data to use.')  
    parser.add_argument('-c', '--checkpoint_path',
                        type=str, 
                        help='(Required) path to the vanilla 3D Gaussian Splatting Checkpoint to load.')
    parser.add_argument('-i', '--iteration_to_load', 
                        type=int, default=7000, 
                        help='iteration to load.')
    
    # Regularization for coarse SuGaR
    parser.add_argument('-r', '--regularization_type', type=str,
                        help='(Required) Type of regularization to use for coarse SuGaR. Can be "sdf", "density" or "dn_consistency". ' 
                        'We recommend using "dn_consistency" for the best mesh quality.')
    
    # Extract mesh
    parser.add_argument('-l', '--surface_level', type=float, default=0.3, 
                        help='Surface level to extract the mesh at. Default is 0.3')
    parser.add_argument('-v', '--n_vertices_in_mesh', type=int, default=1_000_000, 
                        help='Number of vertices in the extracted mesh.')
    parser.add_argument('--project_mesh_on_surface_points', type=str2bool, default=True, 
                        help='If True, project the mesh on the surface points for better details.')
    parser.add_argument('-b', '--bboxmin', type=str, default=None, 
                        help='Min coordinates to use for foreground.')  
    parser.add_argument('-B', '--bboxmax', type=str, default=None, 
                        help='Max coordinates to use for foreground.')
    parser.add_argument('--center_bbox', type=str2bool, default=True, 
                        help='If True, center the bbox. Default is False.')
    
    # Parameters for refined SuGaR
    parser.add_argument('-g', '--gaussians_per_triangle', type=int, default=1, 
                        help='Number of gaussians per triangle.')
    parser.add_argument('-f', '--refinement_iterations', type=int, default=15_000, 
                        help='Number of refinement iterations.')
    
    # (Optional) Parameters for textured mesh extraction
    parser.add_argument('-t', '--export_uv_textured_mesh', type=str2bool, default=True, 
                        help='If True, will export a textured mesh as an .obj file from the refined SuGaR model. '
                        'Computing a traditional colored UV texture should take less than 10 minutes.')
    parser.add_argument('--square_size',
                        default=8, type=int, help='Size of the square to use for the UV texture.')
    parser.add_argument('--postprocess_mesh', type=str2bool, default=False, 
                        help='If True, postprocess the mesh by removing border triangles with low-density. '
                        'This step takes a few minutes and is not needed in general, as it can also be risky. '
                        'However, it increases the quality of the mesh in some cases, especially when an object is visible only from one side.')
    parser.add_argument('--postprocess_density_threshold', type=float, default=0.1,
                        help='Threshold to use for postprocessing the mesh.')
    parser.add_argument('--postprocess_iterations', type=int, default=5,
                        help='Number of iterations to use for postprocessing the mesh.')
    
    # (Optional) PLY file export
    parser.add_argument('--export_ply', type=str2bool, default=True,
                        help='If True, export a ply file with the refined 3D Gaussians at the end of the training. '
                        'This file can be large (+/- 500MB), but is needed for using the dedicated viewer. Default is True.')
    
    # (Optional) Default configurations
    parser.add_argument('--low_poly', type=str2bool, default=False, 
                        help='Use standard config for a low poly mesh, with 200k vertices and 6 Gaussians per triangle.')
    parser.add_argument('--high_poly', type=str2bool, default=False,
                        help='Use standard config for a high poly mesh, with 1M vertices and 1 Gaussians per triangle.')
    parser.add_argument('--refinement_time', type=str, default=None, 
                        help="Default configs for time to spend on refinement. Can be 'short', 'medium' or 'long'.")
      
    # Evaluation split
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')

    # Output path
    parser.add_argument('-o', '--output_path',
                        type=str, default=None,
                        help='(Optional) path to the output directory.')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    parser.add_argument('--white_background', type=str2bool, default=False, help='Use a white background instead of black.')

    # Parse arguments
    args = parser.parse_args()
    if args.low_poly:
        args.n_vertices_in_mesh = 200_000
        args.gaussians_per_triangle = 6
        print('Using low poly config.')
    if args.high_poly:
        args.n_vertices_in_mesh = 1_000_000
        args.gaussians_per_triangle = 1
        print('Using high poly config.')
    if args.refinement_time == 'short':
        args.refinement_iterations = 2_000
        print('Using short refinement time.')
    if args.refinement_time == 'medium':
        args.refinement_iterations = 7_000
        print('Using medium refinement time.')
    if args.refinement_time == 'long':
        args.refinement_iterations = 15_000
        print('Using long refinement time.')
    refined_mesh_path = None
    if args.export_uv_textured_mesh:
        print('Will export a UV-textured mesh as an .obj file.')
    if args.export_ply:
        print('Will export a ply file with the refined 3D Gaussians at the end of the training.')

    # ----- Resolve output directories -----
    if args.output_path is not None:
        scene_name = args.scene_path.split("/")[-1] or args.scene_path.split("/")[-2]
        coarse_output_dir       = os.path.join(args.output_path, "coarse",       scene_name)
        coarse_mesh_output_dir  = os.path.join(args.output_path, "coarse_mesh",  scene_name)
        refined_output_dir      = os.path.join(args.output_path, "refined",      scene_name)
        refined_mesh_output_dir = os.path.join(args.output_path, "refined_mesh", scene_name)
    else:
        coarse_output_dir       = None
        coarse_mesh_output_dir  = None
        refined_output_dir      = None
        refined_mesh_output_dir = None

    # ----- Optimize coarse SuGaR -----
    coarse_args = AttrDict({
        'checkpoint_path': args.checkpoint_path,
        'scene_path': args.scene_path,
        'iteration_to_load': args.iteration_to_load,
        'output_dir': coarse_output_dir,
        'eval': args.eval,
        'estimation_factor': 0.2,
        'normal_factor': 0.2,
        'gpu': args.gpu,
        'white_background': args.white_background,
    })
    if args.regularization_type == 'sdf':
        coarse_sugar_path = coarse_training_with_sdf_regularization(coarse_args)
    elif args.regularization_type == 'density':
        coarse_sugar_path = coarse_training_with_density_regularization(coarse_args)
    elif args.regularization_type == 'dn_consistency':
        coarse_sugar_path = coarse_training_with_density_regularization_and_dn_consistency(coarse_args)
    else:
        raise ValueError(f'Unknown regularization type: {args.regularization_type}')
    
    
    # ----- Extract mesh from coarse SuGaR -----
    coarse_mesh_args = AttrDict({
        'scene_path': args.scene_path,
        'checkpoint_path': args.checkpoint_path,
        'iteration_to_load': args.iteration_to_load,
        'coarse_model_path': coarse_sugar_path,
        'surface_level': args.surface_level,
        'decimation_target': args.n_vertices_in_mesh,
        'project_mesh_on_surface_points': args.project_mesh_on_surface_points,
        'mesh_output_dir': coarse_mesh_output_dir,
        'bboxmin': args.bboxmin,
        'bboxmax': args.bboxmax,
        'center_bbox': args.center_bbox,
        'gpu': args.gpu,
        'eval': args.eval,
        'use_centers_to_extract_mesh': False,
        'use_marching_cubes': False,
        'use_vanilla_3dgs': False,
    })
    coarse_mesh_path = extract_mesh_from_coarse_sugar(coarse_mesh_args)[0]
    
    
    # ----- Refine SuGaR -----
    refined_args = AttrDict({
        'scene_path': args.scene_path,
        'checkpoint_path': args.checkpoint_path,
        'mesh_path': coarse_mesh_path,      
        'output_dir': refined_output_dir,
        'iteration_to_load': args.iteration_to_load,
        'normal_consistency_factor': 0.1,    
        'gaussians_per_triangle': args.gaussians_per_triangle,        
        'n_vertices_in_fg': args.n_vertices_in_mesh,
        'refinement_iterations': args.refinement_iterations,
        'bboxmin': args.bboxmin,
        'bboxmax': args.bboxmax,
        'export_ply': args.export_ply,
        'eval': args.eval,
        'gpu': args.gpu,
        'white_background': args.white_background,
    })
    refined_sugar_path = refined_training(refined_args)
    
    
    # ----- Extract mesh and texture from refined SuGaR -----
    if args.export_uv_textured_mesh:
        refined_mesh_args = AttrDict({
            'scene_path': args.scene_path,
            'iteration_to_load': args.iteration_to_load,
            'checkpoint_path': args.checkpoint_path,
            'coarse_mesh_dir': coarse_mesh_output_dir,
            'refined_model_path': refined_sugar_path,
            'mesh_output_dir': refined_mesh_output_dir,
            'n_gaussians_per_surface_triangle': args.gaussians_per_triangle,
            'square_size': args.square_size,
            'eval': args.eval,
            'gpu': args.gpu,
            'postprocess_mesh': args.postprocess_mesh,
            'postprocess_density_threshold': args.postprocess_density_threshold,
            'postprocess_iterations': args.postprocess_iterations,
        })
        refined_mesh_path = extract_mesh_and_texture_from_refined_sugar(refined_mesh_args)

    if args.output_path is not None:
        manifest = {
            "scene_path": os.path.abspath(args.scene_path),
            "vanilla_checkpoint_path": os.path.abspath(args.checkpoint_path),
            "iteration_to_load": args.iteration_to_load,
            "refinement_iterations": args.refinement_iterations,
            "coarse_model_path": os.path.abspath(coarse_sugar_path),
            "coarse_mesh_path": os.path.abspath(coarse_mesh_path),
            "refined_model_path": os.path.abspath(refined_sugar_path),
            "textured_mesh_path": (
                os.path.abspath(refined_mesh_path) if refined_mesh_path is not None else None
            ),
            "texture_square_size": args.square_size,
            "eval": args.eval,
        }
        manifest_path = os.path.join(args.output_path, "run_manifest.json")
        os.makedirs(args.output_path, exist_ok=True)
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)
        print("Run manifest saved at:", manifest_path)
        