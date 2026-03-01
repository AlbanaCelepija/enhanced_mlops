from digitalhub_runtime_hera.dsl import step
from hera.workflows import DAG, Workflow


def baseline_pipeline():
    with Workflow(entrypoint="dag") as w:
        with DAG(name="dag"):
            A = step(template={"action": "job"},
                     function="data_preprocessing",
                     outputs=["dataset"])
            B = step(template={"action": "job", "inputs": {"data": "{{inputs.parameters.data}}"}},
                     function="train-classifier",
                     inputs={"data": A.get_parameter("dataset")})
            A >> B
    return w
     
     
def fairness_pipeline():
    with Workflow(entrypoint="dag") as w:
        with DAG(name="dag"):
            A = step(template={"action": "job"},
                     function="data_preprocessing",
                     outputs=["dataset"])
            B = step(template={"action": "job", "inputs": {"data": "{{inputs.parameters.data}}"}},
                     function="train-classifier",
                     inputs={"data": A.get_parameter("dataset")})
            A >> B
    return w