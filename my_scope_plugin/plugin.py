from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    from .pipelines.bloom_pipeline import BloomPipeline

    register(BloomPipeline)
