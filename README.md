# <h1 align="center" id="title">Code Supplement to: *Unveiling the Hidden Lake-Rich Landscapes Under Earth’s Glaciers*</h1>

This repository contains the Python code used for the glacier thickness inversion and lake mapping analyses presented in *Unveiling the Hidden Lake-Rich Landscapes Under Earth’s Glaciers* (Frank et al., 2025). Specifically, it performs the inversion including calibration of regional parameters and calculates the locations and depths of lakes that would form if glaciers were to melt completely.  

To run the code, clone the repository and ensure that the dependencies listed below are installed.

---

## Quick Start

1. **Clone the repository**  
   ```bash
   git clone git@github.com:hahohe1892/Frank_thk_inv.git
   cd Frank_thk_inv
   
2. **Install the Instructed Glacier Model (IGM)**  
   Follow the [IGM installation instructions](https://github.com/instructed-glacier-model/igm/wiki/1.-Installation).  

3. **Run a thickness inversion example**  
   Navigate to the folder `Inversion/code/` and run:  
   ```bash
   python optimizer.py
   ```

   Note that this will take several hours to run (see below).
   
4. **Run a lake mapping example (can be done without running step 3)**  
  Navigate to the `Lakes` folder and run:  
   ```bash  
    python map_lakes.py
   ```  
    Outputs will be saved as **NetCDF** files (for inversions) or **tif** files (for lakes).

---
## Thickness Inversion

The glacier thickness inversion is performed by the script `Inversion/code/topg_inv.py`, which is a module of the Instructed Glacier Model (IGM; Jouvet and Cordonnier, 2023). For this study, [IGMv2.2.1](https://github.com/instructed-glacier-model/igm/releases/tag/v2.2.1) was used.  

- Installation instructions: [IGM Wiki  Installation](https://github.com/instructed-glacier-model/igm/wiki/1.-Installation)  
- Running instructions: [IGM Wiki  Running IGM](https://github.com/instructed-glacier-model/igm/wiki/3.-Runing-IGM)  

Beyond IGM and its dependencies, the Bayesian calibration requires the [bayesian-optimization package](https://github.com/bayesian-optimization/BayesianOptimization) (Nogueira, 2014) to be installed. The script `Inversion/code/prepro.py` handles preprocessing and is also a module of IGM, while `Inversion/code/loop_script.sh` is a bash script to do the inversion for several glaciers sequentially.

The script `Inversion/code/optimizer.py` is a wrapper that perfoms Bayesian calibration of model parameters for a set of glaciers by iteratively running the thickness inversion and minimizing a cost function. The cost function is eq. (4) in Frank et al. (2025). To run the example provided, follow these steps:

After successful installation of IGM and `bayesian-optimization`, navigate to `Inversion/code`. The necessary input data for five Greenlandic glaciers with thickness observations is prepared and stored under `Inversion/Input_data`. Fixed model parameters are defined in `Inversion/code/params.json`.  

From within the folder, run:
```bash
python optimizer.py
```

This performs the following steps, optimizing the model parameters ice viscosity `A`, sliding coefficient `c`, regularization parameter `theta`and velocity multiplier `vel_Mult`:
1. Runs the inversion for all five glaciers using defined initial parameters.
2. Stores output **NetCDF** files in `Inversion/Output_data/`
3. Calculates the cost function eq.(4) based on the differences in observed and modeled thicknesses and velocities.
4. Suggests a new parameter combination to probe such that the likelihood of overlooking a minimum in the cost function is minimized.
5. Runs the inversion with these new parameters
6. Repeats steps 2 - 5 `n=5` times.
7. Stores the final best combination of parameters and all other tested parameters combinations in `Inversion/code/optimization_logs.log`.
   
Note that `n=5` is chosen here to prevent excessive computations. In Frank et al. (2025), `n>=30`.

To only run the inversion for one combination of parameters, manipulate the parameters in `Inversion/code/loop_script.sh` and do
```
./loop_script.sh
```

To run own inversions with Bayesian parameter calibration, simply provide own input files in `Inversion/Input_data/` observing the naming convention (`foo_input.nc` where _foo_ is the name of a given glacier) and the structure of the provided `netCDF` files. 

### Input Data
The input data provided was sourced from two resources, both of them providing acces to or being themselves derived from other published data sets:
-  the [OGGM shop](https://docs.oggm.org/en/stable/shop.html) (Maussion et al., 2019).  
- the **Python Glacier Evolution Model (PyGEM)** (Rounce et al., 2023; [GitHub](https://github.com/PyGEM-Community/PyGEM), [Docs](https://pygem.readthedocs.io/en/latest/index.html)) for mass balance forcing.  

To generate custom input data, refer to these resources or others and observe the structure of the provided `foo_input.nc` files.  

### Output
The output is stored as a **NetCDF** file in the run folder. It includes:  
- Initial conditions  
- Intermediate inversion steps  
- Final result at the last time step  

---

## Lake Mapping

Lakes in the subglacial topography are identified using a sink-fill algorithm (Zhou et al., 2016) implemented in [RichDEM](https://richdem.readthedocs.io/en/latest/).  

To run the provided example, install `RichDEM`, then navigate to the `Lakes` folder and execute:
```bash
python map_lakes.py
```
To apply the code to other subglacial topographies, provide a list of file paths in `map_lakes.py`.

---

## Hardware Requirements

The inversion runs efficiently on GPUs and was tested on an **NVIDIA A40 GPU (48 GB VRAM)** with a runtime of ~x hours.  
Note: Memory requirements may exceed the capacity of smaller GPUs.

The lake mapping runs within seconds on the provided examples.

---

## References

1. Frank, T., van Pelt, W. J. J., Rounce, D. R., Jouvet, G., Hock, R.: *Unveiling the Hidden Lake-Rich Landscapes Under Earth’s Glaciers*, in review, 2025.  
2. Jouvet, G., Cordonnier, G.: Ice-flow model emulator based on physics-informed deep learning, *Journal of Glaciology*, 115, https://doi.org/10.1017/jog.2023.73, 2023.
3. Nogueira, F.: Bayesian Optimization: Open source constrained global optimization tool for Python, 2014.
4. Maussion, F., Butenko, A., Champollion, N., Dusch, M., Eis, J., Fourteau, K., Gregor, P., Jarosch, A. H., Landmann, J., Oesterle, F., Recinos, B., Rothenpieler, T., Vlug, A., Wild, C. T., Marzeion, B.: The Open Global Glacier Model (OGGM) v1.1, *Geoscientific Model Development*, 12, 909931, https://doi.org/10.5194/gmd-12-909-2019, 2019.  
5. Rounce, D. R., Hock, R., Maussion, F., Hugonnet, R., Kochtitzky, W., Huss, M., Berthier, E., Brinkerhoff, D., Compagno, L., Copland, L., Farinotti, D., Menounos, B., McNabb, R. W.: Global glacier change in the 21st century: Every increase in temperature matters, *Science*, 379, 7883, https://doi.org/10.1126/science.abo1324, 2023.
6. Zhou, G., Sun, Z., and Fu, S.: An efficient variant of the Priority-Flood algorithm for filling depressions in raster digital elevation models, Computers & Geosciences, 90, 8796, https://doi.org/10.1016/j.cageo.2016.02.021, 2016.
