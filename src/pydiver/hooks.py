# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Project hooks."""
from typing import Any, Dict, Iterable, Optional
import logging
logger = logging.getLogger(__name__)
import os
from pathlib import Path

from kedro.config import ConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.pipeline.node import Node
from kedro.io import DataCatalog
from kedro.versioning import Journal


class NestedParamsHook:
    @hook_impl
    def before_node_run(
        self, catalog: DataCatalog, inputs: Dict[str, Any], run_id: str
    ) -> None:
        pass #import IPython ; IPython.embed() ; exit(1)



class CreateDatasetFoldersHook:
    @staticmethod
    @hook_impl
    def after_catalog_created(catalog, conf_catalog, conf_creds, feed_dict, save_version, load_versions, run_id):
        entries = catalog.list()
        for entry in entries:
            try:
                dset = getattr(catalog.datasets, entry)

                if hasattr(dset, "_path"):
                    _make_dirs(dset._path)
                elif hasattr(dset, "_filepath"):
                    _make_dirs(os.path.split(dset._filepath)[0])
                else:
                    pass
            except AttributeError:
                pass


def _make_dirs(path_to_make):
    if not os.path.exists(path_to_make):
        logger.info(f"Creating missing path {path_to_make}")
        os.makedirs(path_to_make)
    # creates a .gitkeep file while we're at it
    #Path(os.path.join(path_to_make, ".gitkeep")).touch()


class ProjectHooks:
    @hook_impl
    def register_config_loader(
        self, conf_paths: Iterable[str], env: str, extra_params: Dict[str, Any],
    ) -> ConfigLoader:
        #import IPython ; IPython.embed() ; exit(1)
        return ConfigLoader(conf_paths)

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version, journal
        )
