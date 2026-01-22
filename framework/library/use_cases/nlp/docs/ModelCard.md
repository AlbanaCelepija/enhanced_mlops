
# Parameter Efficient Fine-Tuning (PEFT)
PEFT freezes the foundation model and adds a small number of new parameters to learn a specific task.
This method is most commonly associated with LoRA.

## The Goal
Reduce VRAM requirements and storage space, while maintaining performance close to full fine-tuning.

## Architectural Change

Adding adapter modules or low-rank matrices (LoRA) alongside the existing layers.

## Parameter Changes

Trains the parameters in new modules added to the architecture.

## Major Techniques

* LoRA (Low-Rank Adaptation): Adds trainable rank-decomposition matrices to each layer while keeping the original parameters frozen.
* QLoRA (Quantized Low-Rank Adaptation): A variant of LoRA. Compresses the foundation model down to 4-bit precision.
* Adapters: Inserts small, new bottleneck layers between existing transformer blocks.
* Prefix Tuning: Adds trainable prefix tensors to the hidden states of every layer.

## Use Cases

Multi-tenant applications where a single foundation model serves dozens of different task-specific adapters for different users.
E.g., Different departmens of public administration. Adds an adapter trained on each department archives to the foundation model. Every time the public servant requires the model service, their corresponding adapter is loaded to serve them.