from a3_exp.lib import Experiment
from a3_exp.strategies import CMAES
from ariel.simulation.environments import OlympicArena


class CheckOlympic(Experiment):

    def main(self):
        self.init_nde_hpd(42)
        es1 = CMAES(64 * 3, 100, seed=42)
        res1 = es1.ask()


        print(res1[0])

        print(OlympicArena().spec.to_xml())


CheckOlympic().main()
