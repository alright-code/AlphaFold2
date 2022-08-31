# AlphaFold2
PyTorch Implementation of a Simplified AlphaFold2 for Spring 2022 CSCI6969 at RPI

https://magazine.rpi.edu/sharing-your-building-blocks

## AlphaFold2 Paper and Supplementary Information this is Based From
https://www.nature.com/articles/s41586-021-03819-2

https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf

## Notes
- Training is done through `src/train.py`, along with parameter configuration
- See the supplementary information document for parameter information and implementation details
- Code was tested on a very scaled down version of the complete model, giving less than optimal results
- Special thanks to einops and einsum
- TODO: Half-Precision Training, Complete Model Train/Test
