import os
import sys
import torch
import torch.distributed as dist

class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()

def bench_kineto(fn, kernel_names, num_tests: int = 30,
                 suppress_kineto_output: bool = False,
                 trace_path: str = None,flush_l2: bool = True,
                 with_multiple_kernels: bool = False):
    # Conflict with Nsight Systems
    using_nsys = os.environ.get('MO_NSYS_PROFILING', 0)

    # By default, flush L2 with a reasonable size memset to give the GPU some chill time
    flush_l2_size = int(1e9 // 4)  # 减少到1GB，避免显存不足

    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output and not using_nsys else empty_suppress
    with suppress():
        #一个warmup阶段一个采集阶段
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1) if not using_nsys else None
        if not using_nsys:
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule)
        else:
            profiler = empty_suppress()
            
        with profiler:
            for i in range(2):
                for _ in range(num_tests):
                    if flush_l2 and i == 0:
                        try:
                            torch.empty(int(flush_l2_size), dtype=torch.int, device='cuda').zero_()
                        except RuntimeError:
                            # 如果显存不足，跳过 L2 flush
                            pass
                    fn()
                
                if not using_nsys:
                    profiler.step()

    # Return 1 if using Nsight Systems
    if using_nsys:
        return 1

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = profiler.key_averages().table(sort_by='cuda_time_total', max_name_column_width=300).split('\n')
    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    
    if not with_multiple_kernels:
        for name in kernel_names:
            assert sum([name in line for line in prof_lines]) == 1, f'Errors of the kernel {name} in the profiling table: {prof_lines}'

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return average kernel times
    units = {'ms': 1e3, 'us': 1e6}
    kernel_times = []
    for name in kernel_names:
        total_time = 0
        total_num = 0
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                num_str = line.split()[-1]
                for unit, scale in units.items():
                    if unit in time_str:
                        total_time += float(time_str.replace(unit, '')) / scale * int(num_str)
                        total_num += int(num_str)
                        break
        kernel_times.append(total_time / total_num)

    return tuple(kernel_times) if is_tuple else kernel_times[0]


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total

