# Combined Compression and Quality Metrics

Color Legend: 🟢 Best → 🟡 Average → 🔴 Worst

| codec | filesize_ratio | compression_time_sec | decompression_time_sec | psnr_mean | ssim_mean | lpips_mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| jpegxl_lossless | 🔴 0.4811 | 🔴 1140.4031 | 🔴 40.3422 | 🔴 inf | 🟢 1.0000 | 🟢 0.0000 |
| jpegxl_lossy_decompression_1 | 🟢 0.0312 | 🔴 1130.3032 | 🟢 11.9842 | 🔴 60.5440 | 🟢 0.9992 | 🟢 0.0038 |
| jpegxl_lossy_decompression_3 | 🟢 0.0313 | 🔴 1112.0025 | 🟢 10.4759 | 🔴 60.5103 | 🟢 0.9992 | 🟢 0.0036 |
| jpegxl_lossy_decompression_5 | 🟢 0.0365 | 🔴 1053.8656 | 🟡 23.6609 | 🔴 60.4353 | 🟢 0.9992 | 🟢 0.0030 |
| jpegxl_lossy_effort_1 | 🟢 0.0280 | 🟢 752.2647 | 🟢 11.7633 | 🔴 58.7934 | 🟢 0.9987 | 🟢 0.0048 |
| jpegxl_lossy_effort_3 | 🟢 0.0261 | 🟢 751.9134 | 🟢 13.7266 | 🔴 58.7934 | 🟢 0.9987 | 🟢 0.0048 |
| jpegxl_lossy_effort_5 | 🟢 0.0305 | 🔴 1124.7426 | 🟢 12.6818 | 🔴 60.5447 | 🟢 0.9992 | 🟢 0.0038 |
| jpegxl_lossy_hmq | 🟢 0.0176 | 🔴 1132.2740 | 🟢 13.5133 | 🔴 57.5030 | 🟢 0.9983 | 🟢 0.0106 |
| jpegxl_lossy_hq | 🟢 0.0305 | 🔴 1129.6467 | 🟢 12.6491 | 🔴 60.5447 | 🟢 0.9992 | 🟢 0.0038 |
| jpegxl_lossy_lq | 🟢 0.0071 | 🔴 1127.0092 | 🟡 22.1623 | 🔴 52.3416 | 🔴 0.9948 | 🔴 0.0555 |
| jpegxl_lossy_mlq | 🟢 0.0088 | 🔴 1124.0523 | 🟡 22.7159 | 🔴 53.4452 | 🔴 0.9959 | 🔴 0.0422 |
| jpegxl_lossy_mq | 🟢 0.0118 | 🔴 1121.1329 | 🟡 16.6795 | 🔴 55.2753 | 🟡 0.9972 | 🟡 0.0217 |
| lz4hc | 🔴 0.6282 | 🔴 1121.7081 | 🟢 2.1285 | 🔴 inf | 🟢 1.0000 | 🟢 0.0000 |
| zlib | 🔴 0.6073 | 🔴 1174.8494 | 🟢 6.6577 | 🔴 inf | 🟢 1.0000 | 🟢 0.0000 |
| zstd | 🔴 0.5949 | 🔴 1157.1907 | 🟢 3.0154 | 🔴 inf | 🟢 1.0000 | 🟢 0.0000 |

## Metrics Explanation

- **filesize_ratio**: Compressed size / raw size (lower is better)
- **compression_time_sec**: Time to compress all images (lower is better)
- **decompression_time_sec**: Time to decompress all images (lower is better)
- **psnr_mean**: Peak Signal-to-Noise Ratio in dB (higher is better, 30+ good, 35+ excellent)
- **ssim_mean**: Structural Similarity Index 0-1 (higher is better, 0.9+ good)
- **lpips_mean**: Learned Perceptual similarity (lower is better, <0.1 good)
