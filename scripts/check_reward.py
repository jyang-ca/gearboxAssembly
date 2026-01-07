import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_latest_reward(log_dir):
    event_files = glob.glob(os.path.join(log_dir, "summaries/events.out.tfevents.*"))
    if not event_files:
        return None
    
    latest_file = max(event_files, key=os.path.getmtime)
    print(f"Reading file: {latest_file}")
    
    event_acc = EventAccumulator(latest_file)
    event_acc.Reload()
    
    all_tags = event_acc.Tags()
    tags = all_tags.get('scalars', [])
    
    results = {}
    for tag in tags:
        try:
            events = event_acc.Scalars(tag)
            if events:
                first_vals = [e.value for e in events[:min(10, len(events))]]
                last_vals = [e.value for e in events[-min(10, len(events)):]]
                avg_first = sum(first_vals) / len(first_vals)
                avg_last = sum(last_vals) / len(last_vals)
                results[tag] = {
                    'first': avg_first,
                    'last': avg_last,
                    'count': len(events),
                    'trend': avg_last - avg_first,
                    'relative_change': (avg_last - avg_first) / (abs(avg_first) + 1e-9)
                }
        except:
            continue
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "./runs/LongTrajectoryAssembly_approach_20260105_161050"
    
    if os.path.isdir(target):
        log_path = target
    else:
        log_path = os.path.dirname(target)
        if 'summaries' in target:
             log_path = os.path.dirname(log_path)
    
    data = get_latest_reward(log_path)
    if data:
        print("\nSummary of Metrics:")
        for tag, metrics in sorted(data.items()):
            if any(p in tag for p in ['loss', 'kl', 'reward', 'success', 'fps']):
                print(f"Tag: {tag}")
                print(f"  Count: {metrics['count']}")
                print(f"  Initial: {metrics['first']:.6f} -> Latest: {metrics['last']:.6f}")
                print(f"  Change: {metrics['trend']:.6f} ({metrics['relative_change']:.2%})")
    else:
        print("No metrics found.")
