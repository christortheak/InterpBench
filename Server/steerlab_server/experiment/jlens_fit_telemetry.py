"""Best-effort fitting measurements; never changes the estimator or admission."""
import importlib.metadata
import sys
import time

from . import diagnostic_archives as archives


def host_peak_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value * 1024)
    except (ImportError, OSError, ValueError):
        return None


def package_versions():
    result = {}
    for name in ('flash-linear-attention', 'causal-conv1d'):
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = None
    return result


class Measurements:
    def __init__(self, torch, device, path):
        self.torch, self.device, self.path = torch, device, path
        self.cuda = str(device).startswith('cuda') and torch.cuda.is_available()
        self.rows = 0
        self.peak_allocated = self.peak_reserved = None
        self.last = None
        from .runtime_hardware import describe
        self.hardware = describe(torch, device)

    def memory(self):
        if not self.cuda: return None
        try:
            free, capacity = self.torch.cuda.mem_get_info(self.device)
            return dict(freeBytes=int(free), capacityBytes=int(capacity),
                        allocatedBytes=int(self.torch.cuda.memory_allocated(self.device)),
                        reservedBytes=int(self.torch.cuda.memory_reserved(self.device)),
                        peakAllocatedBytes=int(self.torch.cuda.max_memory_allocated(self.device)),
                        peakReservedBytes=int(self.torch.cuda.max_memory_reserved(self.device)))
        except (RuntimeError, ValueError): return None

    def wrap(self, kernel):
        def measured(*args, **kwargs):
            initial = self.memory()
            if initial:
                self.peak_allocated = max(self.peak_allocated or 0, initial['peakAllocatedBytes'])
                self.peak_reserved = max(self.peak_reserved or 0, initial['peakReservedBytes'])
            if self.cuda:
                self.torch.cuda.synchronize(self.device)
                self.torch.cuda.reset_peak_memory_stats(self.device)
            before = self.memory()
            started = time.perf_counter()
            event = dict(fittedOrdinal=self.rows + 1, dimBatch=kwargs['dim_batch'], device=self.device,
                         memoryBefore=before, status='failed')
            try:
                result = kernel(*args, **kwargs)
                if self.cuda: self.torch.cuda.synchronize(self.device)
                event.update(status='fitted', tokens=result[1], validPositions=result[2])
                self.rows += 1
                return result
            finally:
                event.update(seconds=time.perf_counter() - started, memoryAfter=self.memory(),
                             peakHostRSSBytes=host_peak_bytes())
                self.last = event
                memory = event['memoryAfter']
                if memory:
                    self.peak_allocated = max(self.peak_allocated or 0, memory['peakAllocatedBytes'])
                    self.peak_reserved = max(self.peak_reserved or 0, memory['peakReservedBytes'])
                # A failed telemetry write must not hide the numerical exception.
                try:
                    with self.path.open('ab') as stream: stream.write(archives.encoded(event) + b'\n')
                except OSError: pass
        return measured

    def report(self):
        memory = self.memory()
        if memory:
            self.peak_allocated = max(self.peak_allocated or 0, memory['peakAllocatedBytes'])
            self.peak_reserved = max(self.peak_reserved or 0, memory['peakReservedBytes'])
        return dict(device=self.device, hardware=self.hardware, deviceMemory=memory, peakDeviceAllocatedBytes=self.peak_allocated,
                    peakDeviceReservedBytes=self.peak_reserved, peakHostRSSBytes=host_peak_bytes(),
                    rowMeasurementsFile=self.path.name, measuredRows=self.rows,
                    lastMeasurement=self.last, memoryScope='CUDA allocation peaks reset before each fitted row; host RSS is the process lifetime peak.')


class KernelObservation:
    """Observe executed attention modules without selecting or replacing kernels.

    A legacy Transformers module can explicitly disable its fast path. Otherwise
    report the bound implementation and leave dispatch unknown; installed package
    metadata alone cannot establish that a fused kernel actually executed.
    """
    def __init__(self, model):
        import inspect
        self.records, self.handles = {}, []
        for layer in model.layers:
            for module in layer.modules():
                owner = inspect.getmodule(type(module))
                names = getattr(getattr(type(module).forward, '__code__', None), 'co_names', ())
                candidates = [name for name in names if 'gated_delta_rule' in name or 'causal_conv1d' in name]
                if not candidates or owner is None: continue
                def observed(current, inputs, owner=owner, candidates=candidates):
                    available = getattr(owner, 'is_fast_path_available', None)
                    bindings = {name: getattr(getattr(owner, name, None), '__module__', 'unknown') for name in candidates}
                    key = type(current).__module__ + '.' + type(current).__name__
                    self.records[key] = dict(fastPathAvailable=available if type(available) is bool else None,
                                             path='fallback' if available is False else 'unverified', bindings=bindings)
                self.handles.append(module.register_forward_pre_hook(observed))

    def report(self):
        return {'executedModules': self.records, 'note': 'Bindings are observed on executed modules. Only an explicit disabled fast path establishes fallback; other dispatch remains unverified.'}

    def close(self):
        for handle in self.handles: handle.remove()
