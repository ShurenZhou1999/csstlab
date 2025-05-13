## readme

I modify the real space module of package `velocileptors` (https://github.com/sfschen/velocileptors/blob/master/velocileptors/LPT/cleft_fftw.py), to calculate the 1-loop prediction of LPT basis spectra. 

I also include partial codes from `Zenbu` (https://github.com/sfschen/ZeNBu/blob/main/ZeNBu/zenbu.py) to calculate the linear prediction of LPT basis spectra.

The k-expanded version of 1-loop spectra is directly modified from the full resummation module, since it is very easy to realize. I do not use the `*kexpanded` modules in `velocileptors`. 

I have included the basis spectra for Lagrangian fields $\nabla^2\delta$ and $\delta^3$ in the code, for all the 1-loop spectra, the linear spectra, and the k-expanded 1-loop spectra.

