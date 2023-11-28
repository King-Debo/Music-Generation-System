# Import the required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from data import Data
from model import Generator, Discriminator, Critic

# Define some constants
EPOCHS = 100 # The number of epochs for the training
MODEL_DIR = "model" # The directory where the model files will be stored
MODEL_FILE = "model.pth" # The file name for the model file
VALIDATION_FILE = "validation.txt" # The file name for the validation file

# Define a function to compute the gradient penalty for the discriminator
def gradient_penalty(discriminator, real_tensor, fake_tensor):
  # Get the batch size from the real tensor
  batch_size = real_tensor.size(0)
  # Generate a random tensor of shape (batch_size, 1) between 0 and 1
  alpha = torch.rand(batch_size, 1)
  # Expand the alpha tensor to the shape of the real tensor
  alpha = alpha.expand_as(real_tensor)
  # Compute the interpolated tensor by linearly interpolating between the real tensor and the fake tensor
  interpolated_tensor = alpha * real_tensor + (1 - alpha) * fake_tensor
  # Pass the interpolated tensor through the discriminator
  interpolated_output = discriminator(interpolated_tensor)
  # Compute the gradients of the interpolated output with respect to the interpolated tensor
  gradients = torch.autograd.grad(outputs=interpolated_output, inputs=interpolated_tensor, grad_outputs=torch.ones(interpolated_output.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]
  # Flatten the gradients to a vector
  gradients = gradients.view(batch_size, -1)
  # Compute the gradient norm for each sample
  gradient_norm = gradients.norm(2, dim=1)
  # Compute the gradient penalty by averaging the squared difference between the gradient norm and 1
  gradient_penalty = ((gradient_norm - 1) ** 2).mean()
  # Return the gradient penalty
  return gradient_penalty

# Define a function to compute the entropy regularization for the generator
def entropy_regularization(generator, output_tensor):
  # Get the batch size and the output size from the output tensor
  batch_size, output_size = output_tensor.size()
  # Reshape the output tensor to a matrix of shape (batch_size * output_size, 1)
  output_tensor = output_tensor.view(-1, 1)
  # Pass the output tensor through a softmax layer
  output_tensor = nn.Softmax(dim=1)(output_tensor)
  # Compute the entropy for each sample
  entropy = -torch.sum(output_tensor * torch.log(output_tensor + EPSILON), dim=1)
  # Reshape the entropy to a tensor of shape (batch_size, output_size)
  entropy = entropy.view(batch_size, output_size)
  # Compute the entropy regularization by averaging the entropy
  entropy_regularization = entropy.mean()
  # Return the entropy regularization
  return entropy_regularization

# Define a function to train the model
def train_model():
  # Create an instance of the Data class
  data = Data()
  # Create an instance of the Generator class
  generator = Generator()
  # Create an instance of the Discriminator class
  discriminator = Discriminator()
  # Create an instance of the Critic class
  critic = Critic()
  # Create an instance of the Adam optimizer for the generator
  generator_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2), eps=EPSILON)
  # Create an instance of the Adam optimizer for the discriminator
  discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2), eps=EPSILON)
  # Create an instance of the Adam optimizer for the critic
  critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2), eps=EPSILON)
  # Create an instance of the data loader
  data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
  # Loop for a number of epochs
  for epoch in range(EPOCHS):
    # Initialize the generator loss, the discriminator loss, and the critic loss to zero
    generator_loss = 0
    discriminator_loss = 0
    critic_loss = 0
    # Loop for a number of batches
    for batch in data_loader:
      # Get the input tensors and the input parameters from the batch
      input_tensors, input_parameters = batch
      # Feed the input tensors and the input parameters to the generator to get the output tensors
      output_tensors = generator(input_tensors, input_parameters)
      # Feed the output tensors and the input tensors to the discriminator to get the discriminator outputs
      discriminator_outputs = discriminator(output_tensors, input_tensors)
      # Feed the output tensors, the input tensors, and the input parameters to the critic to get the critic outputs
      critic_outputs = critic(output_tensors, input_tensors, input_parameters)
      # Compute the generator loss as the negative mean of the discriminator outputs plus the gamma times the entropy regularization
      generator_loss = -torch.mean(discriminator_outputs) + GAMMA * entropy_regularization(generator, output_tensors)
      # Compute the discriminator loss as the negative mean of the discriminator outputs for the real tensors minus the mean of the discriminator outputs for the fake tensors plus the lambda times the gradient penalty
      discriminator_loss = -torch.mean(discriminator(input_tensors)) + torch.mean(discriminator_outputs) + LAMBDA * gradient_penalty(discriminator, input_tensors, output_tensors)
      # Compute the critic loss as the negative mean of the critic outputs
      critic_loss = -torch.mean(critic_outputs)
      # Update the generator parameters using the generator optimizer
      generator_optimizer.zero_grad()
      generator_loss.backward()
      generator_optimizer.step()
      # Update the discriminator parameters using the discriminator optimizer
      discriminator_optimizer.zero_grad()
      discriminator_loss.backward()
      discriminator_optimizer.step()
      # Update the critic parameters using the critic optimizer
      critic_optimizer.zero_grad()
      critic_loss.backward()
      critic_optimizer.step()
      # Print the generator loss, the discriminator loss, and the critic loss
      print(f"Generator loss: {generator_loss.item():.4f}, Discriminator loss: {discriminator_loss.item():.4f}, Critic loss: {critic_loss.item():.4f}")
    # Save the model parameters to a file
    torch.save(generator.state_dict(), os.path.join(MODEL_DIR, MODEL_FILE))
    # Test the model on a validation set and print the performance metrics
    test_model()
