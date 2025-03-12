def next_step_prediction_task(data, model, device):
    inputs = data[0]
    targets = data[1]

    inputs = inputs.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs, feature = model(inputs)  # Forward pass through the model
    outputs = outputs + inputs
    return (inputs, outputs, targets)


def reconstruction_task(data, model, device):
    inputs = data

    inputs = inputs.to(device)
    targets = inputs
    outputs, feature = model(inputs)

    return (inputs, outputs, targets)