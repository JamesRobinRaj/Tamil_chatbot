# Tamil_chatbot
This project leverages the capabilities of the "Hemanth-thunder/Tamil-Mistral-7B-Instruct-v0.1" model to perform text generation in Tamil. Here's a breakdown of what the project does:

Model Setup: It uses the transformers library to load a pre-trained language model optimized for Tamil text. The model is configured with quantization options to improve performance on available hardware.

Language Detection and Translation: The project uses the langdetect library to detect the language of the input text. If the input text is not in Tamil, it uses the translate library to translate the text into Tamil before processing it with the model.

Text Generation: The project employs a text generation pipeline to generate responses based on the input prompts. It uses a template to format the prompts, ensuring the generated output is coherent and relevant to the input query.

Interactive Queries: The project is set up to handle different queries, translating non-Tamil input into Tamil, and then generating responses. It demonstrates this with examples such as solving arithmetic expressions, asking about well-being, and storytelling.

Overall, the project aims to facilitate interaction with a Tamil language model, allowing users to input queries in any language and receive responses in Tamil after translation and processing.
