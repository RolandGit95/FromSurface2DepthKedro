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
        
        

"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from pydiver.pipelines import data_simulation as ds
from pydiver.pipelines import data_processing as dp
from pydiver.pipelines import model_validation as mv

        
def register_pipelines() -> Dict[str, Pipeline]:

    #data_simulation_pipeline_A = ds.create_pipeline(regime="A")
    #data_processing_pipeline_A = dp.create_pipeline(regime="A")
    model_validation_pipeline_A = mv.create_pipeline(regime="A")

    
    #data_simulation_pipeline_B = ds.create_pipeline(regime="B")
    #data_processing_pipeline_B = dp.create_pipeline(regime="B")
    #model_validation_pipeline_B = mv.create_pipeline(regime="B")
    #data_scient_pipeline = ds.create_pipeline()

    
    return {
        #"__default__": data_simulation_pipeline_A + data_processing_pipeline_A + model_validation_pipeline_A,
        "__default__": model_validation_pipeline_A,

        #"ds_A": data_simulation_pipeline_A,
        #"dp_A": data_processing_pipeline_A,
        "mv_A": model_validation_pipeline_A,

        #"ds_B": data_simulation_pipeline_B,
        #"dp_B": data_processing_pipeline_B,
        #"mv_B": model_validation_pipeline_B,
            }











