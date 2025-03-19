import sys, os
import numpy as np
from ..base import BaseEmulator_GP
from ..emu_simu import Emulator_simu
from ..emu_loop import Emulator_loop



_PathData = os.path.dirname(os.path.dirname(__file__)) + "/data/"
_FileLoop = _PathData + "GP_loop.npy"
_FileLin  = _PathData + "GP_lin.npy"
_EmuLoop = Emulator_loop(kmax=1.05)
_EmuLin  = Emulator_loop(kmax=1.05, N_PCs=12)

try : _EmuLoop._load_emulator( _FileLoop )
except:
    raise FileNotFoundError("Read the train the `Template` loop-power-sepctrum Emulator and save the parameters. ")

try : _EmuLin._load_emulator( _FileLin  )
except:
    raise FileNotFoundError("Read the train the `linear scale` loop-power-sepctrum Emulator and save the parameters. ")





class PerformanceTest(BaseEmulator_GP):
    def __init__(self, ):
        super().__init__()
        pass
    def _load_emulator(self, **kwargs):
        pass
    def _train_emulator(self, **kwargs):
        pass



class LeaveOneOut(PerformanceTest):
    def __init__(self, ):
        super().__init__()
        self.__PathData = _PathData
        self.__kmax = 1.05
        self.__Nz = 12
        self.EmuSimu = Emulator_simu(kmax=self.__kmax)
        self.__load_raw_results()
        self.__use_reFit_theory()

    def __load_raw_results(self, ):
        Dload = np.load( self.__PathData + "HLPT_simulation_Pkij.npy", allow_pickle=True, )[()]
        self.Parameters = Dload["Parameters"]
        self.redshifts = Dload["redshifts"]
        self.Pk_D = Dload["Pkij"]
        self.Pk_T = Dload["Pkij_T"]
        self.k = Dload["k"]
        self.Ncosmo = self.Parameters.shape[0]

        self.krange = self.k < self.__kmax
        self.k = self.k[self.krange]
        self.Pk_D = self.Pk_D[..., self.krange]
        self.Pk_T = self.Pk_T[..., self.krange]   # this will be replaced. 
    
    def __use_reFit_theory(self, ):
        paramsNorm = self.NormalizeParam(self.Parameters )
        self.Pk_T = np.array([ _EmuLoop( iparam ) for iparam in paramsNorm ]) 
    

    def LOO(self, drop):
        '''
        Parameters
        ----------
        drop : int, the ID of cosmology dropped from the training set
        '''
        if drop >= self.Ncosmo : return None
        X_drop = self.Parameters[drop]
        X = np.vstack([
            self.Parameters[ :drop], 
            self.Parameters[ drop+1 :]
        ])
        X_drop = self.To_Abscissas(self.NormalizeParam(X_drop))
        X = self.To_Abscissas_MultiCosmo(self.NormalizeParam(X))
        pk_D_ = np.vstack([ 
            self.Pk_D[ :drop ] , 
            self.Pk_D[ drop+1 :], 
        ])
        pk_T_ = np.vstack([ 
            self.Pk_T[ :drop ] , 
            self.Pk_T[ drop+1 :], 
        ])
        self.EmuSimu._train_emulator(X, self.k, pk_D_, pk_T_, to_save=False, )
        y_pred = self.EmuSimu(X_drop, self.Pk_T[drop] ,)
        y_true = self.Pk_D[drop]
        
        return y_pred, y_true
    

    def LOO_all(self, ):
        pk_pred = [ ]
        pk_true = [ ]
        for drop in range(self.Ncosmo):
            p1, p2 = self.LOO(drop)
            pk_pred.append(p1)
            pk_true.append(p2)
        pk_pred, pk_true = np.array(pk_pred), np.array(pk_true)
        return pk_pred, pk_true

    




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------


class LeavePartOut(PerformanceTest):
    def __init__(self, set=1, ):
        super().__init__()
        self.__PathData = _PathData
        self.set = set
        self.__kmax = 1.05
        self.__Nz = 12
        if set==1:
            self.EmuLoop = _EmuLoop  # Emulator_loop(kmax=self.__kmax)
        elif set==2:
            self.EmuLoop = _EmuLin
        self.__load_raw_results()

    def __load_raw_results(self, ):
        if self.set == 1:
            Eload = np.load( self.__PathData + "HLPT_1loop_calculation.npy", allow_pickle=True, )[()]
            self.ext_Pkij_T = Eload["Pkij_T"]
        elif self.set == 2:
            Eload = np.load( self.__PathData + "HLPT_1loop_calculation_linear-scale.npy", allow_pickle=True, )[()]
            self.ext_Pkij_T = Eload["Pkij"]
        self.ext_Params = Eload["Parameters"]
        
        self.k = Eload["k"]
        self.krange = self.k < self.__kmax
        self.k = self.k[self.krange]
        self.ext_Pkij_T = self.ext_Pkij_T[..., self.krange]

        self.Ncosmo = self.ext_Params.shape[0]
        self.Nloop_samples = _EmuLoop.N_training_samples

    def validate(self, IndexTest=None, ):
        if IndexTest is None :
            test0, test1 = self.Nloop_samples, self.Nloop_samples + 1200
        else:
            test0,  test1 = self.Nloop_samples + IndexTest[0], self.Nloop_samples + IndexTest[1]
        paramsNorm = self.NormalizeParam(self.ext_Params[test0:test1])
        pk_T_test  = self.ext_Pkij_T[test0:test1]
        y_pred = np.array([ self.EmuLoop(ips) for ips in paramsNorm ])
        y_pred_out = np.zeros(pk_T_test.shape)
        for l, (i, j) in enumerate(self._index):
            y_pred_out[..., l, :] = y_pred[..., i, j, :]
        return y_pred_out, pk_T_test




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
