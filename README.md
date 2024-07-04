# LLM data generation

Using the groq API, and subclassing the `pytorch.utils.data.Dataset` class,
we can use an LLM to generate data on the fly, along with a callback function to check validity of the generated text.

## Intention
This is to demonstrate how we can use LLMs for text data augmentation for NLP projects.

## NOTE
This is an example and is not intended to be used right out of the box. You might want to use some pydantic models, A proper API class for groq and it's related callbacks, etc. to make this more clean.
