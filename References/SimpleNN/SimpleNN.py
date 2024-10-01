import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from finn.util.pytorch import ToTensor
from qonnx.core.datatype import DataType
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import brevitas.onnx as bo
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.infer_datatypes import InferDataTypes

# Define your neural network model class (or import it if defined elsewhere)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Adjust the input dimensions according to the training configuration
        self.fc1 = nn.Linear(28*28, 128)  # Assuming the original model was trained on MNIST
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Adjust input size for MNIST
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function
        x = self.fc2(x)  # Apply the second fully connected layer
        return x

# Create an instance of the model
model = SimpleNN()

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Stochastic Gradient Descent optimizer

# Define transformation to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5 and std=0.5
])

# Load the MNIST dataset
train_dataset = MNIST(root='./../data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

# Training loop
model.train()  # Set the model to training mode
for epoch in range(1):  # Run for one epoch (you can increase this)
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        if batch_idx % 100 == 0:  # Print every 100 batches
            print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item()}")

# Save the model (optional)
torch.save(model.state_dict(), 'outputs/simple_nn.pth')

# Load the trained model
cnv = SimpleNN()
cnv.load_state_dict(torch.load("outputs/simple_nn.pth", map_location=torch.device('cpu')))
cnv.eval()

# Export the model using brevitas
input_tensor = torch.randn(1, 1, 28, 28)  # Example input tensor with MNIST dimensions
bo.export_qonnx(cnv, input_tensor, "outputs/simple_nn_export.onnx")

# Load and transform the ONNX model
model = ModelWrapper("outputs/simple_nn_export.onnx")
model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(RemoveStaticGraphInputs())
model.save("outputs/simple_nn_tidy.onnx")

print('Model successfully exported and transformed')


# Load and prepare the core model
model = ModelWrapper("outputs/simple_nn_tidy.onnx")

# Get input shape
global_inp_name = model.graph.input[0].name
ishape = model.get_tensor_shape(global_inp_name)

# Ensure input shape is in the correct format for ToTensor
if isinstance(ishape, list):
    ishape = tuple(ishape)  # Convert list to tuple if needed

# Prepare the ToTensor transformation
totensor_pyt = ToTensor()

# Export preprocessing model to ONNX
chkpt_preproc_name = "outputs/simple_nn_preproc.onnx"
try:
    bo.export_qonnx(totensor_pyt, torch.zeros(ishape), chkpt_preproc_name)  # Use a tensor with the shape
except Exception as e:
    print(f"Error exporting ToTensor model: {e}")
    raise

# Join preprocessing and core model
pre_model = ModelWrapper(chkpt_preproc_name)
model = model.transform(MergeONNXModels(pre_model))

# Add input quantization annotation
global_inp_name = model.graph.input[0].name
model.set_tensor_datatype(global_inp_name, DataType["UINT8"])

# Save the final model
model.save("outputs/simple_nn_with_preproc.onnx")

print('Model with preprocessing successfully exported and transformed')

# postprocessing: insert Top-1 node at the end
model = model.transform(InsertTopK(k=1))
chkpt_name = "outputs/simple_nn_pre_post.onnx"
# tidy-up again
model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(InferDataTypes())
model = model.transform(RemoveStaticGraphInputs())
model.save(chkpt_name)

from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from onnx import AttributeProto, NodeProto
from qonnx.core.modelwrapper import ModelWrapper

test_pynq_board = "Pynq-Z2"
target_clk_ns = 10

from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild

model = ModelWrapper(chkpt_name)
# First, partition the model into streaming dataflow
model = model.transform(CreateDataflowPartition())
# Apply the ZynqBuild transformation
model = model.transform(ZynqBuild(platform="Pynq-Z2", period_ns=10))

# Apply the PYNQ Driver transformation
model = model.transform(MakePYNQDriver("zynq-iodma"))

# Save the final model
model.save("outputs/simple_nn_synth.onnx")
