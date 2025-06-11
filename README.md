This Git repository is to demonstrate the method to implement the multi-agent cross-modal LLM framework for comprehensive pneumonia assessment on chest x-rays.
Our method is based on multiple task-specified AI agents.
Please run the code in the notebook named "LLM_image_predict.ipynb".
The trained classifier is for COVID-19 pneumonia detection, which is in Hugging Face format and is accessible on Hugging Face Hub: https://huggingface.co/stanleyl79/mobilevit_covid19
The trained regressor is for mRALE based pneumonia severity prediction, which is in Hugging Face format and is accessible on Hugging Face Hub: https://huggingface.co/stanleyl79/dinov2_mrale_regressor
Other AI models, such as the GPT-4o, GPT-o1-mini, please use your own API key to access from the OpenAI platform.
The Huggingface models, including mobile ViT, DinoV2, and BiomedCLIP-PubmedBERT can be accessed using the Hugging Face API.
