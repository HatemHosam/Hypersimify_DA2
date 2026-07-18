# Hypersimify_DA2

![Project Screenshot](method.png)

Hypersimify DepthAnythingv2 for High-Resolution Depth Estimation of non-Lambertian Surfaces. A simple training-free domain adaptation method for depth estimation of transparent and mirror objects in indoor scenes.

#### News 8th best runner in NTIRE 2026 Challenge on High-Resolution Depth of non-Lambertian Surfaces

We propose Hypersimify-DA2, an efficient method for high-resolution monocular metric depth estimation from images of specular and transparent surfaces. We simply map the input image distribution to the synthetic image distribution to match the Hypersim dataset domain. We use the pretrained DepthAnythingv2 (DA2), which is fine-tuned on the Hypersim dataset for metric indoor depth estimation. This DA2 model is pretrained to perfectly estimate the depth of transparent and specular objects since the dataset is synthetic and has perfect ground truth for such objects.

The key novelty of our method is implementing a training-free mapping of the input real image domain to the synthetic image domain (HyperSim-like) using a simple Image processing-based pipeline proven to be good enough to map the image domain. 


