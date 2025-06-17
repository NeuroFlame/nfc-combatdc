import logging
import os
from typing import Dict, Any
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.apis.fl_constant import ReservedKey

from utils.types import ComputationParamDTO
from utils.utils import get_output_directory_path, get_computation_parameters
from utils.logger import NvFlareLogger
from . import aggregator_methods as am

class DCCombatAggregator(Aggregator):

    def __init__(self):
        super().__init__()
        # Store results as a dictionary
        self.site_results: Dict[int, Dict[str, Any]] = {}  # Store results as a dictionary
        self.agg_cache: Dict[str, Any] = {}
        self.agg_cache_dir: str = "./temp_agg_cache"
        os.makedirs(self.agg_cache_dir, exist_ok=True)  # succeeds even if directory exists.
        self.logger = None
        pass

    def accept(self, site_result: Shareable, fl_ctx: FLContext) -> bool:
        """
        Accepts a result from a site and stores it for later aggregation.

        This method is called when a client site sends a result. Developers can override this 
        if they need to handle or validate the results differently before storing them.

        :param site_result: The result received from the client site.
        :param fl_ctx: The federated learning context for this run.
        :return: Boolean indicating if the result was successfully accepted.
        """
        site_name = site_result.get_peer_prop(
            key=ReservedKey.IDENTITY_NAME, default=None)
        contribution_round = fl_ctx.get_prop(key="CURRENT_ROUND", default=None)
        
        if contribution_round is None or site_name is None:
            return False  # Could log a warning/error here as well

        if contribution_round not in self.site_results:
            self.site_results[contribution_round] = {}
            
        if self.logger is None:
            logging.info(f'log_level for aggregator: {fl_ctx.get_prop(key="log_level", default=None)}')
            self.logger = NvFlareLogger('aggregator.log', get_output_directory_path(fl_ctx), fl_ctx.get_prop(key="log_level", default="info"))

        # Store the result for the site using its identity name as the key
        self.site_results[contribution_round][site_name] = site_result["result"]

        self.logger.info('accepting site result from: ', site_name, 'from round: ', contribution_round)

        return True
        
    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """
        Aggregates the results from all accepted client sites and produces a global result.

        This is where the global aggregation logic happens. Developers can override this
        if they need to change how the results from each site are combined.

        :param fl_ctx: The federated learning context for this run.
        :return: A Shareable object containing the aggregated global result.
        """
        outgoing_shareable = Shareable()
        contribution_round = fl_ctx.get_prop(key="CURRENT_ROUND", default=None)
        self.logger.info('aggregation round: ', contribution_round)
        try:
            if contribution_round == 0:
                agg_result = am.combat_remote_step1(self.site_results[contribution_round])
                self.agg_cache.update(agg_result['cache'])
                outgoing_shareable['result'] = agg_result['output']
            
            elif contribution_round == 1:
                agg_result = am.combat_remote_step2(self.site_results[contribution_round], self.agg_cache, self.logger)
                self.agg_cache.update(agg_result['cache'])
                outgoing_shareable['result'] = agg_result['output']
            
            elif contribution_round == 2:
                agg_result = am.combat_remote_step3(self.site_results[contribution_round], self.agg_cache, self.logger)
                self.agg_cache.update(agg_result['cache'])
                outgoing_shareable['result'] = agg_result['output']
                self.logger.close()
            return outgoing_shareable
        except Exception as err:
            self.logger.error('Exception: ', err)
            self.logger.close()
            raise Exception(f'exception: {err}')