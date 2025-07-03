## 
<h1 align="center">SUB: Benchmarking CBM Generalization via Synthetic Attribute Substitutions</h1>

<div align="center">
<a href="https://www.eml-munich.de/people/jessica-bader">Jessica Bader</a>,
<a href="https://www.eml-munich.de/people/leander-girrbach">Leander Girrbach</a>,
<a href="https://www.eml-munich.de/people/stephan-alaniz">Stephan Alaniz</a>,
<a href="https://www.eml-munich.de/people/zeynep-akata">Zeynep Akata</a>
<br>
<br>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/TODO)
</div>

<h3 align="center">Abstract</h3>

<p align="justify">
Concept Bottleneck Models (CBMs) and other interpretable models show great promise for making AI applications more transparent, which is essential in fields like medicine. Despite their success, we demonstrate that CBMs struggle to reliably identify the correct concepts under distribution shifts.
To assess the robustness of CBMs to concept variations, we introduce SUB: a fine-grained image and concept benchmark containing 38,400 synthetic images based on the CUB dataset. To create SUB, we select a CUB subset of 33 bird classes and 45 concepts to generate counterfactual images which substitute a specific concept, such as wing color or belly pattern. We introduce a novel Tied Diffusion Guidance (TDG) method to precisely control generated images, where noise sharing for two parallel denoising processes ensures that both the correct bird class and the correct attribute are generated. This novel benchmark enables rigorous evaluation of CBMs and similar interpretable models, contributing to the development of more robust methods.
</p>
<br>
<div align="center">
    <img src="assets/method.svg" alt="Method" width="1000">
</div>

---
### Code
Code will be available soon
### Citation
```bibtex
@article{bader2025sub,
  title={SUB: Benchmarking CBM Generalization via Synthetic Attribute Substitutions}, 
  author={Jessica Bader and Leander Girrbach and Stephan Alaniz and Zeynep Akata},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```