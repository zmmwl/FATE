from fate.interface import Summary as SummaryInterface

class Summary(SummaryInterface):
    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.summary = {}

    def save(self):
        self.ctx.tracker.log_component_summary(summary_data=self.summary)
