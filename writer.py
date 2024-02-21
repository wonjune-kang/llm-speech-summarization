from torch.utils.tensorboard import SummaryWriter


class MyWriter(SummaryWriter):
    def __init__(self, config, logdir):
        super(MyWriter, self).__init__(logdir)
        self.sample_rate = config.audio.sampling_rate

    def log_training(self, losses, step):
        for loss_type, value in losses.items():
            self.add_scalar(f"train/{loss_type}", value, step)

    def log_validation(self, losses, step):
        for loss_type, value in losses.items():
            self.add_scalar(f"validation/{loss_type}", value, step)

    def log_audio_text_responses(self, prompt_audios, prompt_texts, llm_responses, step):
        for i, (audio, text, response) in enumerate(
            zip(prompt_audios, prompt_texts, llm_responses)
        ):
            self.add_audio(f"prompt_audios/audio_{i}", audio, step, self.sample_rate)
            self.add_text(f"prompt_texts/prompt_{i}", text, step)
            self.add_text(f"llm_responses/response_{i}", response, step)
