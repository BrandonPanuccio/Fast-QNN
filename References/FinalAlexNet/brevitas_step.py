from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

model = ModelWrapper('outputs/alexnet_1w1a_mnist_qonnx.onnx')

# Apply the conversion from QONNX to FINN-ONNX
model = model.transform(ConvertQONNXtoFINN())

# Save the converted FINN-ONNX model
model.save('outputs/alexnet_1w1a_mnist_finn.onnx')
print("Model converted to FINN-ONNX format and saved as 'alexnet_1w1a_mnist_finn.onnx'")
