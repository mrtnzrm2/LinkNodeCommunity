from networks.ECoG.Precue.MK1.theta import MK1PreTheta
from networks.ECoG.Precue.MK1.beta import MK1PreBeta
from networks.ECoG.Precue.MK1.highbeta import MK1PreHighBeta
from networks.ECoG.Precue.MK1.gamma import MK1PreGamma

from networks.ECoG.Precue.MK2.theta import MK2PreTheta
from networks.ECoG.Precue.MK2.beta import MK2PreBeta
from networks.ECoG.Precue.MK2.highbeta import MK2PreHighBeta
from networks.ECoG.Precue.MK2.gamma import MK2PreGamma


from networks.ECoG.Postcue.MK1.theta import MK1PostTheta
from networks.ECoG.Postcue.MK1.beta import MK1PostBeta
from networks.ECoG.Postcue.MK1.highbeta import MK1PostHighBeta
from networks.ECoG.Postcue.MK1.gamma import MK1PostGamma

from networks.ECoG.Postcue.MK2.theta import MK2PostTheta
from networks.ECoG.Postcue.MK2.beta import MK2PostBeta
from networks.ECoG.Postcue.MK2.highbeta import MK2PostHighBeta
from networks.ECoG.Postcue.MK2.gamma import MK2PostGamma

WAVES = {
    "MK1PreTheta" : MK1PreTheta,
    "MK1PreBeta" : MK1PreBeta,
    "MK1PreHighBeta" : MK1PreHighBeta,
    "MK1PreGamma" : MK1PreGamma,

    "MK2PreTheta" : MK2PreTheta,
    "MK2PreBeta" : MK2PreBeta,
    "MK2PreHighBeta" : MK2PreHighBeta,
    "MK2PreGamma" : MK2PreGamma,

    "MK1PostTheta" : MK1PostTheta,
    "MK1PostBeta" : MK1PostBeta,
    "MK1PostHighBeta" : MK1PostHighBeta,
    "MK1PostGamma" : MK1PostGamma,
    
    "MK2PostTheta" : MK2PostTheta,
    "MK2PostBeta" : MK2PostBeta,
    "MK2PostHighBeta" : MK2PostHighBeta,
    "MK2PostGamma" : MK2PostGamma
}