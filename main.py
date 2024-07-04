from typing import Callable
from torch.utils.data import Dataset
from torch.utils.data.dataloader import T_co
from dotenv import load_dotenv
from groq import Groq
import os


load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))


class Llamayielder(Dataset):
    def __init__(
        self,
        callback: Callable[..., bool],
        generation_size=10,
    ) -> None:
        self.cache = dict()
        self.callback = callback
        self.generation_size = generation_size
        super().__init__()

    def __len__(self):
        return self.generation_size

    def __getitem__(self, index) -> T_co:
        completion = (
            client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": "Generate a random sentence, each time it has to be a completely different topic",
                    }
                ],
                model="llama3-8b-8192",
            )
            .choices[0]
            .message.content
        )
        # If completion was okay:
        if self.callback(completion):
            if self.cache[index]:
                return self.cache[index]
            else:
                self.cache[index] = completion
                return self.cache[index]
