# MRI Response Prediction
This repository is to store the codes used for conference paper titled: "_Learning-Based MRI Response Predictions from OCT Microvascular Models to Replace Simulation-Based Frameworks_" [DOI: https://doi.org/10.1007/978-3-031-66955-2_4] and journal paper titled "_TBD_" to be published.

## Architectural Modifications

The original AutoEncoder architecture was modified to handle both image features and MRI parameters for predicting MRI signals. The key additions include a dedicated MRI parameter processing branch implemented through self.mri_param_fc, a linear layer that transforms the 3-dimensional MRI parameters into a 64-dimensional feature space. To combine these MRI features with the encoded image features, we added self.combine_features, a 3D convolutional layer that fuses the expanded MRI features with the encoder output. The forward pass was modified to accept both the input image tensor and MRI parameters tensor, with additional processing to properly expand the MRI features to match the spatial dimensions of the encoded image features. For generating the final predictions, we introduced self.final_conv and self.final_fc layers, where the former applies a final 1x1 convolution, and the latter transforms the pooled features into the required output shape of 11 signals with 50 time points each. The network concludes with global average pooling and reshaping operations to produce the final output tensor of shape (batch_size, 11, 50), representing the predicted MRI signals for each input sample.



