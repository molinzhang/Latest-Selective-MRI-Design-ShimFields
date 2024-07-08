# Updated-Selective-MR-Imaging-Joint-Optimization-Design-with-RF-and-Shim-Fields
This repo contains the official code for the MRM papers ["Stochastic‐offset‐enhanced restricted slice excitation and 180° refocusing designs with spatially non‐linear ΔB0 shim array fields"](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.29827) and ["Selective RF excitation designs enabled by time‐varying spatially non‐linear ΔB0 fields with applications in fetal MRI"](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.29114).

This code is cleaned up from the previous repo ["Stochastic-offset-enhanced-restricted-slice-excitation-and-refocusing-designs"](https://github.com/molinzhang/Stochastic-offset-enhanced-restricted-slice-excitation-and-refocusing-designs).

As a recall, the code for the selective MR imaging optimization design is derived from AutoDiffPulses](https://github.com/tianrluo/AutoDiffPulses) which provides the optimization framework for RF pulse and B0 fields (linear gradient fields and shim array fields) with auto-differentiation. We followed the same manner of [AutoDiffPulses](https://github.com/tianrluo/AutoDiffPulses). The actual optimization part is performed with Pytorch and wrapped with MATLAB.
  
Our work enables both excitation and refocusing designs by optimizing Rf pulse and time-varying $\Delta B_0$ shim array fields. Note that we used additional linear gradient fields but fixed them during the optimization. Optimizing linear gradient fields yields worse results. 

Compared with the previous repo ["Stochastic-offset-enhanced-restricted-slice-excitation-and-refocusing-designs"](https://github.com/molinzhang/Stochastic-offset-enhanced-restricted-slice-excitation-and-refocusing-designs)., we added dependecies  `adpulses` and `mrphy` from AutoDiffPulses to our repository. Also, all hyper-parameters can be adjusted at the beginning of the MATLAB file `ex_op.m`.

### Dependencies.

Use `environment.yml` to install required packages.

### Key features for usage.

To run the code, `IniVar.m` and `gpu_id` must be provided for the MATLAB code. We have provided an `IniVar.m` file in this repo. Feel free to upload your own. For more details about this file, please refer to [AutoDiffPulses](https://github.com/tianrluo/AutoDiffPulses). You can also change the directory to the `shim_path`, `rf_path`, `mask_path`, `B0_path`, `B1_path`, ect. There are more explanetory comments of those hyperparameters in the MATLAB file. 

Note that this is an 'ancient' code developed back in 2020, some of the features might not be available in the latest MATLAB or Pytorch. We could run the code with the dependencies in `environment.yml` and on a single NVIDIA TITAN V. 


## Citation

If you find this work helpful, please consider citing the following two papers:

1. "Stochastic‐offset‐enhanced restricted slice excitation and 180° refocusing designs with spatially non‐linear ΔB0 shim array fields."
```bibtex
@article{zhang2023stochastic,
  title={Stochastic-offset-enhanced restricted slice excitation and 180° refocusing designs with spatially non-linear $\Delta$B0 shim array fields},
  author={Zhang, Molin and Arango, Nicolas and Arefeen, Yamin and Guryev, Georgy and Stockmann, Jason P and White, Jacob and Adalsteinsson, Elfar},
  journal={Magnetic Resonance in Medicine},
  volume={90},
  number={6},
  pages={2572--2591},
  year={2023},
  publisher={Wiley Online Library}
}
```

2. "Selective RF excitation designs enabled by time‐varying spatially non‐linear ΔB0 fields with applications in fetal MRI."
```bibtex
@article{zhang2022selective,
  title={Selective RF excitation designs enabled by time-varying spatially non-linear $\Delta$ B 0 fields with applications in fetal MRI},
  author={Zhang, Molin and Arango, Nicolas and Stockmann, Jason P and White, Jacob and Adalsteinsson, Elfar},
  journal={Magnetic Resonance in Medicine},
  volume={87},
  number={5},
  pages={2161--2177},
  year={2022},
  publisher={Wiley Online Library}
}
```



## License

This project is licensed under the [MIT License](LICENSE).
