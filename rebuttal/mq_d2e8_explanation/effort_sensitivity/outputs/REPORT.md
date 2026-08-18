# Approximate effort sensitivity

HQ and E3 both use JPEG XL distance 1 in the archived producer source. HQ omitted the effort argument while E3 set effort 3. The historical numeric default and encoder build are not frozen, so this is an approximate sensitivity—not a controlled effort experiment.

## Fixed-recipe paired bootstrap

| Family | E3 − HQ product (95% interval) | Holm result |
|---|---:|---|
| cp_measure | -0.00228 [-0.00658, +0.00197] | unresolved ($p_{Holm}=0.2883$) |
| DINOv2 | +0.00350 [+0.00032, +0.00659] | unresolved ($p_{Holm}=0.081$) |
| MorphEM | +0.00503 [+0.00143, +0.00895] | E3>HQ ($p_{Holm}=0.02064$) |
| OpenPhenom | +0.00471 [+0.00016, +0.00939] | unresolved ($p_{Holm}=0.0872$) |
| SubCell | +0.00951 [+0.00436, +0.01511] | E3>HQ ($p_{Holm}=0.0006$) |

One Zstd-selected recipe was frozen across codecs. PA (306 compound clusters) and PC (201 target clusters) were independently resampled 50,000 times with shared weights across model/codec columns. Product intervals omit unknown PA–PC covariance.

## Supporting archived evidence

- Pixel metrics use 100 matched sites per codec.
- HQ/E3 median SSIM: 0.999260/0.998785.
- E3 exceeds HQ correlation for 11.8% of 790 cp_measure features.
- Cell and nuclei segmentation comparisons use the exact 9,216-site common set.

## Limitation

The available codecs are not a distance-by-effort factorial: HQ=(D1, default/omitted effort), E3=(D1,E3), D2-E8=(D2,E8), and MQ=(D3, default/omitted effort). This analysis cannot attribute D2-E8 versus MQ behavior to effort.
