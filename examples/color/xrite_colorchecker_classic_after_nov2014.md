# X Rite ColorChecker Classic Reference

The `xrite_colorchecker_classic_after_nov2014` preset supports the 24-patch X-Rite ColorChecker Classic Nano used in the specimen capture. X-Rite introduced the Nano format in 2019, so it uses the November 2014 and newer patch formulation.

The source is X-Rite's `ColorChecker24_After_Nov2014.txt` CGATS file. Its header identifies X-Rite as the manufacturer, the November 2014 and newer edition, M0 measurement conditions, and an i1Pro 2 instrument. The downloaded source file had SHA-256 `2d110e3e8c33cb46fad920817efcf3efa504c2a9eb9cf60c9079b9c111557d29`.

The manufacturer values are CIELAB under D50. They were converted to D65 with Bradford chromatic adaptation and then encoded as sRGB. The preset stores the resulting normalized values in the standard row-major chart order. The UF_Herp_3998 chart is upside down in the reference images, so its configuration sets `target_rotation_degrees` to `180`.

The X-Rite support page explains the pre-November-2014 and post-November-2014 formulations and provides the generic reference files. It also recommends measuring the individual physical chart when the highest possible accuracy is required. The generic manufacturer values are suitable for this project calibration, but they do not replace a chart-specific spectrophotometer measurement.

- X-Rite reference edition guidance: https://www.xrite.com/en/service-support/new_color_specifications_for_colorchecker_sg_and_classic_charts
- X-Rite ColorChecker target brochure: https://www.xrite.com/-/media/xrite/files/literature/l11/l11-200_l11-299/l11-293_colorchecker-targets_en.pdf
- Archived manufacturer CGATS data: https://babelcolor.com/index_htm_files/ColorChecker24_After_Nov2014.txt
