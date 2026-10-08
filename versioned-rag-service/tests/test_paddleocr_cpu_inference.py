from types import SimpleNamespace


def test_cpu_inference_uses_repeatable_native_kernel_configuration():
    from src.paddleocr_quality import configure_cpu_inference
    calls=[]
    fake=SimpleNamespace(set_num_threads=lambda n:calls.append(n),
        backends=SimpleNamespace(mkldnn=SimpleNamespace(enabled=True)))
    configure_cpu_inference(fake)
    assert calls==[1]
    assert fake.backends.mkldnn.enabled is False
