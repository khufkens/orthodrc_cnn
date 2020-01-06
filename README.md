# Ortho DRC Deep Learning forest classifier

This is the code used to train the model (src/forest_model.h5) to classify forest and non forest pixels and a 1958 historical orthomosaic covering a large section of Yangambi in the Central Congo Basin, DR Congo (then Belgian Congo).

When using the model cite the model / code / data as:

`Hufkens et al. 2020. Historical aerial surveys map long-term changes of
forest cover and structure in the central Congo Basin. Submitted.`

For full details on the methodology I refer to the above manuscript, or the documentation provided in [the repository website](https://khufkens.github.io/orthodrc_cnn/).

## Licenses & citation

I'm indebted to the [segmentation models](https://github.com/qubvel/segmentation_models) python package and examples from which I extensively borrowed code. To be inline with licensing all code is distributed under an MIT license. All other (image) data, and the resulting model (forest_model.h5) is distributed under a AGPLv3 license.

## Acknowledgements

This research was supported through the Belgian Science Policy office COBECORE project (BELSPO; grant BR/175/A3/COBECORE) and from the European Union Marie Skłodowska-Curie Action (project number 797668).
