## 
<h1 align="center">SUB: Benchmarking CBM Generalization via Synthetic Attribute Substitutions</h1>

<div align="center">
<a href="https://www.eml-munich.de/people/jessica-bader">Jessica Bader</a>,
<a href="https://www.eml-munich.de/people/leander-girrbach">Leander Girrbach</a>,
<a href="https://www.eml-munich.de/people/stephan-alaniz">Stephan Alaniz</a>,
<a href="https://www.eml-munich.de/people/zeynep-akata">Zeynep Akata</a>
<br>
<br>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2507.23784)
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
### Dataset
SUB is available at https://huggingface.co/datasets/Jessica-bader/SUB.
### TDG Code
Tied Diffusion Guidance (TDG) can be tested with tied_diffusion_guidance_demo.py, by specifying --prompt1 and --prompt2. For example:
```
python tied_diffusion_guidance_demo.py --prompt1 "a blue jay with a yellow crown" --prompt2 "a bird with a yellow crown"
```

### CBM Evaluation Code
An example for evaluating a CBM trained 'independently' can be found in CBM_testing/test_ind_cbm_example.py. Before using this:

(1) Several files from the Concept Bottleneck repository must be imported. The CBM repository can be pulled from https://github.com/yewsiang/ConceptBottleneck, and CBM_testing/test_ind_cbm_example.py can be placed inside.

(2) Train the desired CBM on CUB, as outlined in https://github.com/yewsiang/ConceptBottleneck and saved in /path/to/cbm.pth. 

(3) Download the metadata attributes.txt from the CUB dataset. The parameter ATTRIBUTE_FILE at the top of test_ind_cbm_example.py must point to this file, likely something like: /path/to/CUB_200_2011/attributes/attributes.txt

(4) CBM training on CUB requires a file outlined in https://github.com/yewsiang/ConceptBottleneck as CUB_processed/class_attr_data_10/val.pkl, which can be obtained from the instructions there. Change LABEL_PATH at the top of test_ind_cbm_example.py to point to this file, likely something like: /path/to/CUB_processed/class_attr_data_10/val.pkl.

(5) This code takes the CUB class names from the CUB dataset folders. Change IMG_DIR_PATH at the top of test_ind_cbm_example.py to the path to the CUB images, likely something like: /path/to/CUB_200_2011/images. This is only used for class names.

(6) The provided example loads from a list of SUB files, kept in a .csv file with column names ['file', 'changed_attr', 'bird']. This can be kept at /path/to/sub_data.csv. Alternatively, the code can be modified to use the provided HuggingFace dataset.

Then, it can be run with:
```
python CBM_testing/test_ind_cbm_example.py --model_path /path/to/cbm.pth --output_path /path/to/result/outputs/ind_SUB.txt --file_path /path/to/sub_data.csv
```

--test_for_compliment is an optional flag, to specify that the removed attribute should be evaluated. Standard behavior without this flag will evaluate the SUB-added attribute.

### MLLM Evaluation Code
An example of MLLM evaluation code is provided, implemented on CLIP. First, evaluate CLIP on SUB with CBM_testing/test_clip_example.py. 
```
python CBM_testing/test_clip_example.py --file_path /path/to/sub_data.csv
```
Optionally, --model and --pretrained may be specified, in accordance with open_clip.

This will save a result file to results/menon_vondrick_clip/{args.model}_{args.pretrained}/attribute_clip_our_results.csv. The score can be calculated with CBM_testing/process_clip_outputs_ours.ipynb. 

(1) ATTRIBUTE_FILE and LABEL_PATH must be set, to the same paths as in CBM Evaluation Code. 

(2) Set path_to_baseline_labels to the file saved by python CBM_testing/test_clip_example.py (results/menon_vondrick_clip/{args.model}_{args.pretrained}/attribute_clip_our_results.csv). Set evaluate_a_compliment to False if evaluating the SUB-modified attribute, or True if evaluating the removed attribute. 

The results of this notebook will be printed in the bottom line. 

### Citation
```bibtex
@article{bader2025sub,
  title={SUB: Benchmarking CBM Generalization via Synthetic Attribute Substitutions}, 
  author={Jessica Bader and Leander Girrbach and Stephan Alaniz and Zeynep Akata},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```
