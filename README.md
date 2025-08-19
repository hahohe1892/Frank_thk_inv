# Code supplement to: Unveiling the Hidden Lake-Rich Landscapes Under Earth’s Glaciers

The code provided here demonstrates the key functionality of the thickness inversion as described in Frank et al. (2023) and applied in "Unveiling the Hidden Lake-Rich Landscapes Under Earth’s Glaciers" (Frank et al., 2025). 

It constitutes a module of the Instructed Glacier Model (IGM; Jouvet et al., 2022) v2.2.1 available [here](https://github.com/instructed-glacier-model/igm/releases/tag/v2.2.1). For instructions on how to use IGM, please refer to the documentation in the corresponding repository.

_Importantly, this code only runs a toy example and does not produce output used in Frank et al. (2025)._ \
Specifically, it performs a simple thickness and friction inversion for Storglaciären using made-up mass balance forcing. For it to be used in a meaningful way, use realistic mass balance and dh/dt forcing and ensure that all input datasets are consistent with each other.\
Minor deviations between the code used in Frank et al. (2025) and this code may exist. However, the key principles of the bed and friction inversions are identical and thus this code provides relevant context for intepreting the findings of Frank et al. (2025).


# References
1. Frank, T., van Pelt, W. J. J., Rounce, D.R., Jouvet, G., Hock, R.: Unveiling the Hidden Lake-Rich Landscapes Under Earth’s Glaciers, in review, 2025.
2. Frank, T., van Pelt, W. J. J., and Kohler, J.: Reconciling ice dynamics and bed topography with a versatile and fast ice thickness inversion. The Cryosphere, 17, 4021–4045. https://doi.org/10.5194/tc-17-4021-2023, 2023.
3. Jouvet, G., Cordonnier, G., Kim, B., Lüthi, M., Vieli, A., Aschwanden, A.: Deep learning speeds up ice flow modelling by several orders of magnitude. Journal of Glaciology 68, 651–664. https://doi.org/10.1017/jog.2021.120, 2022
