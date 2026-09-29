# Third-Party Notices and Licenses

This project ("dvlt") is distributed under the Apache License, Version 2.0
(see [LICENSE](LICENSE)). It incorporates portions of code from the
third-party projects listed below. For each project the adapted files,
upstream attribution, and the full license text are reproduced.

The dvlt project also depends on, but does **not** vendor, several other
third-party packages (the upstream baselines used for evaluation). Those are
installed via `pip` by the end user and are governed solely by their own
licenses; they are listed at the end of this document for completeness.

---

# DINOv2

Source: https://github.com/facebookresearch/dinov2

**Adapted files:**
- `src/dvlt/model_components/dinov2.py` — DINOv2 vision-transformer backbone.
- `src/dvlt/model_components/layers/attention.py` — multi-head self-attention.
- `src/dvlt/model_components/layers/block.py` — transformer block, DropPath, LayerScale, Mlp primitives.
- `src/dvlt/model_components/layers/patch_embed.py` — patch embedding.
- `src/dvlt/model_components/layers/swiglu_ffn.py` — SwiGLU feed-forward network.
- `src/dvlt/model_components/layers/rope.py` — 2D Rotary Position Embeddings.

**Pretrained encoder weights:** The dvlt model initializes its DINOv2 encoder
backbone from the public DINOv2 checkpoints hosted at
`dl.fbaipublicfiles.com/dinov2/...`, released under Apache-2.0.

**Upstream copyright:** Copyright (c) Meta Platforms, Inc. and affiliates.

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW OR
      AGREED TO IN WRITING, LICENSOR PROVIDES THE WORK (AND EACH
      CONTRIBUTOR PROVIDES ITS CONTRIBUTIONS) ON AN "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
      IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OR CONDITIONS
      OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
      PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE
      APPROPRIATENESS OF USING OR REDISTRIBUTING THE WORK AND ASSUME ANY
      RISKS ASSOCIATED WITH YOUR EXERCISE OF PERMISSIONS UNDER THIS LICENSE.

   8. Limitation of Liability. IN NO EVENT AND UNDER NO LEGAL THEORY,
      WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE,
      UNLESS REQUIRED BY APPLICABLE LAW (SUCH AS DELIBERATE AND GROSSLY
      NEGLIGENT ACTS) OR AGREED TO IN WRITING, SHALL ANY CONTRIBUTOR BE
      LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL,
      INCIDENTAL, OR CONSEQUENTIAL DAMAGES OF ANY CHARACTER ARISING AS A
      RESULT OF THIS LICENSE OR OUT OF THE USE OR INABILITY TO USE THE
      WORK (INCLUDING BUT NOT LIMITED TO DAMAGES FOR LOSS OF GOODWILL,
      WORK STOPPAGE, COMPUTER FAILURE OR MALFUNCTION, OR ALL OTHER
      COMMERCIAL DAMAGES OR LOSSES), EVEN IF SUCH CONTRIBUTOR HAS BEEN
      ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
```

---

# PyTorch3D

Source: https://github.com/facebookresearch/pytorch3d

**Adapted files:**
- `src/dvlt/common/rotation.py` — functions `quat_to_mat`, `mat_to_quat`,
  `_sqrt_positive_part`, and `standardize_quaternion` are adapted from
  [`pytorch3d.transforms.rotation_conversions`](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py).
  The quaternion convention was switched from PyTorch3D's scalar-first (WXYZ)
  to scalar-last (XYZW) to match the convention used elsewhere in dvlt.
  `so3_relative_angle` and `quaternion_slerp` are new code not part of the
  PyTorch3D originals.

**Upstream copyright:** Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

```
BSD 3-Clause License

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

# Depth-Anything-3

Source: https://github.com/ByteDance-Seed/Depth-Anything-3

**Adapted files:**
- `src/dvlt/common/rays.py` — homography fitting, RANSAC, and ray-to-camera
  helpers adapted from
  [`utils/ray_utils.py`](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/utils/ray_utils.py).

**Upstream copyright:** Copyright (c) 2025 ByteDance Ltd. and/or its affiliates

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW OR
      AGREED TO IN WRITING, LICENSOR PROVIDES THE WORK (AND EACH
      CONTRIBUTOR PROVIDES ITS CONTRIBUTIONS) ON AN "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
      IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OR CONDITIONS
      OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
      PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE
      APPROPRIATENESS OF USING OR REDISTRIBUTING THE WORK AND ASSUME ANY
      RISKS ASSOCIATED WITH YOUR EXERCISE OF PERMISSIONS UNDER THIS LICENSE.

   8. Limitation of Liability. IN NO EVENT AND UNDER NO LEGAL THEORY,
      WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE,
      UNLESS REQUIRED BY APPLICABLE LAW (SUCH AS DELIBERATE AND GROSSLY
      NEGLIGENT ACTS) OR AGREED TO IN WRITING, SHALL ANY CONTRIBUTOR BE
      LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL,
      INCIDENTAL, OR CONSEQUENTIAL DAMAGES OF ANY CHARACTER ARISING AS A
      RESULT OF THIS LICENSE OR OUT OF THE USE OR INABILITY TO USE THE
      WORK (INCLUDING BUT NOT LIMITED TO DAMAGES FOR LOSS OF GOODWILL,
      WORK STOPPAGE, COMPUTER FAILURE OR MALFUNCTION, OR ALL OTHER
      COMMERCIAL DAMAGES OR LOSSES), EVEN IF SUCH CONTRIBUTOR HAS BEEN
      ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
```

---

# AnyCalib

Source: https://github.com/javrtg/AnyCalib

**Adapted files:**
- `src/dvlt/common/rays.py` — S² (unit-sphere) tangent-space helpers
  `s2_geodesic_distance` and `s2_logmap_at_z1`, adapted from
  [`anycalib/manifolds.py`](https://github.com/javrtg/AnyCalib/blob/main/anycalib/manifolds.py)
  (AnyCalib: On-Manifold Learning for Model-Agnostic Single-View Camera
  Calibration, ICCV 2025).

**Upstream copyright:** Copyright (c) 2025 Javier Tirado-Garín (AnyCalib authors)

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW OR
      AGREED TO IN WRITING, LICENSOR PROVIDES THE WORK (AND EACH
      CONTRIBUTOR PROVIDES ITS CONTRIBUTIONS) ON AN "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
      IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OR CONDITIONS
      OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
      PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE
      APPROPRIATENESS OF USING OR REDISTRIBUTING THE WORK AND ASSUME ANY
      RISKS ASSOCIATED WITH YOUR EXERCISE OF PERMISSIONS UNDER THIS LICENSE.

   8. Limitation of Liability. IN NO EVENT AND UNDER NO LEGAL THEORY,
      WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE,
      UNLESS REQUIRED BY APPLICABLE LAW (SUCH AS DELIBERATE AND GROSSLY
      NEGLIGENT ACTS) OR AGREED TO IN WRITING, SHALL ANY CONTRIBUTOR BE
      LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL,
      INCIDENTAL, OR CONSEQUENTIAL DAMAGES OF ANY CHARACTER ARISING AS A
      RESULT OF THIS LICENSE OR OUT OF THE USE OR INABILITY TO USE THE
      WORK (INCLUDING BUT NOT LIMITED TO DAMAGES FOR LOSS OF GOODWILL,
      WORK STOPPAGE, COMPUTER FAILURE OR MALFUNCTION, OR ALL OTHER
      COMMERCIAL DAMAGES OR LOSSES), EVEN IF SUCH CONTRIBUTOR HAS BEEN
      ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
```

---

# MoGe

Source: https://github.com/microsoft/MoGe

**Adapted files:**
- `src/dvlt/model_components/pose_encoding.py` — `create_uv_grid` is a direct
  port of MoGe's `normalized_view_plane_uv` helper
  ([`moge/utils/geometry_torch.py`](https://github.com/microsoft/MoGe/blob/main/moge/utils/geometry_torch.py)).

**Upstream copyright:** Copyright (c) Microsoft Corporation.

```
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

# MultiNeRF

Source: https://github.com/google-research/multinerf

**Adapted files:**
- `src/dvlt/common/geometry.py` — `_compute_residual_and_jacobian` and
  `radial_and_tangential_undistort` adapted from
  [`internal/camera_utils.py`](https://github.com/google-research/multinerf/blob/main/internal/camera_utils.py).

**Upstream copyright:** Copyright 2023 Google LLC

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW OR
      AGREED TO IN WRITING, LICENSOR PROVIDES THE WORK (AND EACH
      CONTRIBUTOR PROVIDES ITS CONTRIBUTIONS) ON AN "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
      IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OR CONDITIONS
      OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
      PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE
      APPROPRIATENESS OF USING OR REDISTRIBUTING THE WORK AND ASSUME ANY
      RISKS ASSOCIATED WITH YOUR EXERCISE OF PERMISSIONS UNDER THIS LICENSE.

   8. Limitation of Liability. IN NO EVENT AND UNDER NO LEGAL THEORY,
      WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE,
      UNLESS REQUIRED BY APPLICABLE LAW (SUCH AS DELIBERATE AND GROSSLY
      NEGLIGENT ACTS) OR AGREED TO IN WRITING, SHALL ANY CONTRIBUTOR BE
      LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL,
      INCIDENTAL, OR CONSEQUENTIAL DAMAGES OF ANY CHARACTER ARISING AS A
      RESULT OF THIS LICENSE OR OUT OF THE USE OR INABILITY TO USE THE
      WORK (INCLUDING BUT NOT LIMITED TO DAMAGES FOR LOSS OF GOODWILL,
      WORK STOPPAGE, COMPUTER FAILURE OR MALFUNCTION, OR ALL OTHER
      COMMERCIAL DAMAGES OR LOSSES), EVEN IF SUCH CONTRIBUTOR HAS BEEN
      ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
```

---

# VGGT

Source: https://github.com/facebookresearch/vggt

**Adapted files**:

| File | Derived from |
|---|---|
| `src/dvlt/model_components/loss/util.py` | `training/loss.py` (`torch_quantile`, `point_map_to_normal`, `check_and_fix_inf_nan`, `filter_by_quantile`) and `training/train_utils/general.py` |
| `src/dvlt/model_components/loss/common.py` | `training/loss.py` (`gradient_loss`, `normal_loss`, `gradient_loss_multi_scale_wrapper`, confidence-weighted regression formula) |
| `src/dvlt/model_components/loss/camera.py` | `training/loss.py` (`camera_loss_single`, `compute_camera_loss` — pose-encoding split, deep-supervision weighting, valid-frame mask) |
| `src/dvlt/data/sampler.py` | `training/data/dynamic_dataloader.py` — dynamic batching design (`image_num`/`aspect_ratio` API, `max_frames // image_num` batch sizing) |

Per VGGT License §1(b)(ii), any publication of research using this codebase
must acknowledge use of VGGT.

**Upstream copyright:** Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

```
VGGT License

v1 Last Updated: July 29, 2025

"Acceptable Use Policy" means the Acceptable Use Policy, applicable to Research Materials, that is incorporated into this Agreement.

"Agreement" means the terms and conditions for use, reproduction, distribution and modification of the Research Materials set forth herein.

"Documentation" means the specifications, manuals and documentation accompanying Research Materials distributed by Meta.

"Licensee" or "you" means you, or your employer or any other person or entity (if you are entering into this Agreement on such person or entity's behalf), of the age required under applicable laws, rules or regulations to provide legal consent and that has legal authority to bind your employer or such other person or entity if you are entering in this Agreement on their behalf.

"Meta" or "we" means Meta Platforms Ireland Limited (if you are located in or, if you are an entity, your principal place of business is in the EEA or Switzerland) and Meta Platforms, Inc. (if you are located outside of the EEA or Switzerland).

"Research Materials" means, collectively, Documentation and the models, software and algorithms, including machine-learning model code, trained model weights, inference-enabling code, training-enabling code, fine-tuning enabling code, demonstration materials and other elements of the foregoing distributed by Meta and made available under this Agreement.

By clicking "I Accept" below or by using or distributing any portion or element of the Research Materials, you agree to be bound by this Agreement.

1. License Rights and Redistribution.

a. Grant of Rights. You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license under Meta's intellectual property or other rights owned by Meta embodied in the Research Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the Research Materials.

b. Redistribution and Use.

i. Distribution of Research Materials, and any derivative works thereof, are subject to the terms of this Agreement. If you distribute or make the Research Materials, or any derivative works thereof, available to a third party, you may only do so under the terms of this Agreement. You shall also provide a copy of this Agreement to such third party.

ii. If you submit for publication the results of research you perform on, using, or otherwise in connection with Research Materials, you must acknowledge the use of Research Materials in your publication.

iii. Your use of the Research Materials must comply with applicable laws and regulations (including Trade Control Laws) and adhere to the Acceptable Use Policy, which is hereby incorporated by reference into this Agreement.

2. User Support. Your use of the Research Materials is done at your own discretion; Meta does not process any information nor provide any service in relation to such use. Meta is under no obligation to provide any support services for the Research Materials. Any support provided is "as is", "with all faults", and without warranty of any kind.

3. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW, THE RESEARCH MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, AND META DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE RESEARCH MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE RESEARCH MATERIALS AND ANY OUTPUT AND RESULTS.

4. Limitation of Liability. IN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY DIRECT OR INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.

5. Intellectual Property.

a. Subject to Meta's ownership of Research Materials and derivatives made by or for Meta, with respect to any derivative works and modifications of the Research Materials that are made by you, as between you and Meta, you are and will be the owner of such derivative works and modifications.

b. If you institute litigation or other proceedings against Meta or any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Research Materials, outputs or results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other rights owned or licensable by you, then any licenses granted to you under this Agreement shall terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold harmless Meta from and against any claim by any third party arising out of or related to your use or distribution of the Research Materials.

6. Term and Termination. The term of this Agreement will commence upon your acceptance of this Agreement or access to the Research Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein. Meta may terminate this Agreement if you are in breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete and cease use of the Research Materials. Sections 5, 6 and 9 shall survive the termination of this Agreement.

7. Governing Law and Jurisdiction. This Agreement will be governed and construed under the laws of the State of California without regard to choice of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement. The courts of California shall have exclusive jurisdiction of any dispute arising out of this Agreement.

8. Modifications and Amendments. Meta may modify this Agreement from time to time; provided that they are similar in spirit to the current version of the Agreement, but may differ in detail to address new problems or concerns. All such changes will be effective immediately. Your continued use of the Research Materials after any modification to this Agreement constitutes your agreement to such modification. Except as provided in this Agreement, no modification or addition to any provision of this Agreement will be binding unless it is in writing and signed by an authorized representative of both you and Meta.

Acceptable Use Policy

Meta seeks to further understanding of new and existing research domains with the mission of advancing the state-of-the-art in artificial intelligence through open research for the benefit of all.

As part of this mission, Meta makes certain research materials available for use in accordance with this Agreement (including the Acceptable Use Policy). Meta is committed to promoting the safe and responsible use of such research materials.

Prohibited Uses

You agree you will not use, or allow others to use, Research Materials to:

1. Violate the law or others' rights, including to:
   - Engage in, promote, generate, contribute to, encourage, plan, incite, or further illegal or unlawful activity or content, such as violence or terrorism, exploitation or harm to children, human trafficking, sexual solicitation, or any other criminal activity.
   - Engage in, promote, incite, or facilitate the harassment, abuse, threatening, or bullying of individuals or groups of individuals.
   - Engage in, promote, incite, or facilitate discrimination or other unlawful or harmful conduct in the provision of employment, employment benefits, credit, housing, other economic benefits, or other essential goods and services.
   - Engage in the unauthorized or unlicensed practice of any profession including, but not limited to, financial, legal, medical/health, or related professional practices.
   - Collect, process, disclose, generate, or infer health, demographic, or other sensitive personal or private information about individuals without rights and consents required by applicable laws.
   - Engage in or facilitate any action or generate any content that infringes, misappropriates, or otherwise violates any third-party rights, including the outputs or results of any technology using Research Materials.
   - Create, generate, or facilitate the creation of malicious code, malware, computer viruses or do anything else that could disable, overburden, interfere with or impair the proper working, integrity, operation or appearance of a website or computer system.

2. Engage in, promote, incite, facilitate, or assist in the planning or development of activities that present a risk of death or bodily harm to individuals, including use of research artifacts related to military, warfare, nuclear industries or applications, espionage, guns and illegal weapons, illegal drugs and regulated/controlled substances, operation of critical infrastructure, transportation technologies, or heavy machinery, self-harm or harm to others, or any content intended to incite or promote violence, abuse, or any infliction of bodily harm to an individual.

3. Intentionally deceive or mislead others, including use of Research Materials related to generating, promoting, or furthering fraud or disinformation, defamatory content, spam, impersonating another individual without consent, representing that outputs of research materials are human-generated, or generating or facilitating false online engagement.

4. Fail to appropriately disclose to end users any known dangers of your Research Materials.
```

---

## Upstream packages used for evaluation (installed separately by the user)

These projects are imported by dvlt's evaluation wrappers
(`src/dvlt/model/{da3,vggt,vggt_omega,mapanything,pi3}/model.py`). They are
**not vendored** in this repository; the user installs them with `pip`. Each
is governed by its own license.

| Package | Wrapper module | License | Install |
|---|---|---|---|
| [`facebookresearch/vggt`](https://github.com/facebookresearch/vggt) | `dvlt.model.vggt.model` | See upstream `LICENSE.txt`. Code: commercial-use-friendly. Checkpoints: original is non-commercial; `VGGT-1B-Commercial` permits commercial use. | `pip install git+https://github.com/facebookresearch/vggt.git --no-deps` |
| [`facebookresearch/vggt-omega`](https://github.com/facebookresearch/vggt-omega) | `dvlt.model.vggt_omega.model` | FAIR Noncommercial Research License (non-commercial / research-only; code + checkpoints). See upstream `LICENSE`. | `pip install git+https://github.com/facebookresearch/vggt-omega.git --no-deps` |
| [`ByteDance-Seed/Depth-Anything-3`](https://github.com/ByteDance-Seed/Depth-Anything-3) | `dvlt.model.da3.model` | Apache-2.0 (`Copyright 2025 The Depth Anything 3 Team`) | `pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git --no-deps` |
| [`facebookresearch/map-anything`](https://github.com/facebookresearch/map-anything) | `dvlt.model.mapanything.model` | Apache-2.0 | `pip install git+https://github.com/facebookresearch/map-anything.git --no-deps` |
| Pi3 / Pi3X (upstream repo) | `dvlt.model.pi3.model` | Non-commercial only (CC BY-NC-SA 4.0 on vendored DUSt3R-derived files; non-commercial repo header). | see `docs/INSTALL.md` |

---

## License summary

The majority of this project's **code** is released under the **Apache License,
Version 2.0**. The **model weights** (the `nvidia/dvlt` checkpoint) are not
covered by Apache-2.0; they are released under the **NVIDIA License**
(non-commercial; research and evaluation use only), see
`LICENSES/NVIDIA-LICENSE.txt`.

The following four files are derived from VGGT (Meta Platforms, Inc.) and are
distributed under the **VGGT License** (see `LICENSES/VGGT-LICENSE.txt`):

- `src/dvlt/model_components/loss/util.py`
- `src/dvlt/model_components/loss/common.py`
- `src/dvlt/model_components/loss/camera.py`
- `src/dvlt/data/sampler.py`

All other inbound licenses (Apache-2.0, BSD-3-Clause, MIT) are compatible with
the project's outbound Apache-2.0 license. Each adapted file in `src/dvlt/`
carries the upstream copyright notice in its top-of-file header, satisfying
§4(a–c) of the Apache License, §1 of the MIT License, and the redistribution
conditions of the BSD-3-Clause License as applicable.
