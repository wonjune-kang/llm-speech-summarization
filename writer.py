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

    def log_validation_perplexity(self, perplexity, input_type, step):
        self.add_scalar(f"validation/{input_type}_perplexity", perplexity, step)

    def log_audio_text_responses(
        self, prompt_audios, prompt_texts, audio_responses, text_responses, epoch
    ):
        for i, (audio, text, audio_response, text_response) in enumerate(
            zip(prompt_audios, prompt_texts, audio_responses, text_responses)
        ):
            self.add_audio(f"prompt_audios/audio_{i}", audio, epoch, self.sample_rate)
            self.add_text(f"prompt_texts/prompt_{i}", text, epoch)
            self.add_text(f"llm_audio_responses/response_{i}", audio_response, epoch)
            self.add_text(f"llm_text_responses/response_{i}", text_response, epoch)
