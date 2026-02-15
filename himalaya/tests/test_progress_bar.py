from himalaya.progress_bar import bar
from himalaya.progress_bar import ProgressBar
from himalaya.progress_bar import _format_time
import time
import sys
import io


def test_progress_bar():
    # simple smoke test
    for ii in bar(range(10), title="La barre"):
        pass

    bar_ = ProgressBar(title="La barre", max_value=10, initial_value=0,
                       max_chars=40, progress_character='.', spinner=False,
                       verbose_bool=True)
    for ii in bar_(range(10)):
        pass

    bar_ = ProgressBar(max_value=10, title="La barre")
    for ii in range(10):
        bar_.update_with_increment_value(1)
    bar_.close()


def test_format_time_seconds():
    """Test _format_time with various second values."""
    # Test 0 seconds
    assert _format_time(0) == "00:00:00"
    
    # Test single digit seconds
    assert _format_time(5) == "00:00:05"
    
    # Test double digit seconds
    assert _format_time(45) == "00:00:45"
    
    # Test 59 seconds (edge case before minute)
    assert _format_time(59) == "00:00:59"


def test_format_time_minutes():
    """Test _format_time with minute values."""
    # Test 1 minute exactly
    assert _format_time(60) == "00:01:00"
    
    # Test 1 minute 30 seconds
    assert _format_time(90) == "00:01:30"
    
    # Test 5 minutes 45 seconds
    assert _format_time(345) == "00:05:45"
    
    # Test 59 minutes 59 seconds (edge case before hour)
    assert _format_time(3599) == "00:59:59"


def test_format_time_hours():
    """Test _format_time with hour values."""
    # Test 1 hour exactly
    assert _format_time(3600) == "01:00:00"
    
    # Test 1 hour 30 minutes
    assert _format_time(5400) == "01:30:00"
    
    # Test 2 hours 15 minutes 30 seconds
    assert _format_time(8130) == "02:15:30"
    
    # Test 10 hours
    assert _format_time(36000) == "10:00:00"
    
    # Test 24 hours
    assert _format_time(86400) == "24:00:00"
    
    # Test 99 hours 59 minutes 59 seconds
    assert _format_time(359999) == "99:59:59"


def test_format_time_fractional_seconds():
    """Test _format_time with fractional seconds (should truncate)."""
    # Test that fractional seconds are truncated, not rounded
    assert _format_time(1.9) == "00:00:01"
    assert _format_time(59.9) == "00:00:59"
    assert _format_time(60.5) == "00:01:00"
    assert _format_time(3599.9) == "00:59:59"


def test_progress_bar_eta_format():
    """Test that progress bar displays ETA in hh:mm:ss format."""
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        # Create a progress bar
        bar = ProgressBar(title='Test', max_value=100, verbose_bool=True)
        
        # Simulate some progress
        time.sleep(0.1)
        bar.update(10)
        
        # Get the output
        output = sys.stdout.getvalue()
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Check that output contains hh:mm:ss format (two colons)
        assert output.count(':') >= 2, "ETA should contain hh:mm:ss format with at least 2 colons"
        
        # Check that "ETA:" is present when not at 100%
        if '10%' in output and '100%' not in output:
            assert 'ETA:' in output, "ETA should be displayed when progress is not complete"
        
        # Check that the format looks like hh:mm:ss (matches pattern like 00:00:05)
        import re
        eta_pattern = r'ETA: \d{2}:\d{2}:\d{2}'
        assert re.search(eta_pattern, output), f"ETA should match hh:mm:ss format. Output: {output}"
        
    finally:
        # Ensure stdout is always restored
        sys.stdout = old_stdout


def test_progress_bar_eta_at_completion():
    """Test that progress bar doesn't show ETA at completion."""
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        # Create a progress bar and complete it
        bar = ProgressBar(title='Test', max_value=100, verbose_bool=True)
        time.sleep(0.1)
        bar.update(100)
        
        # Get the output
        output = sys.stdout.getvalue()
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # At 100%, there should be no ETA, but there should be it/s
        assert 'it/s' in output, "Should show iteration rate at completion"
        # The last update at 100% should not have ETA
        lines = output.strip().split('\r')
        last_line = lines[-1] if lines else ""
        assert 'ETA:' not in last_line, "Should not show ETA at 100% completion"
        
    finally:
        sys.stdout = old_stdout


def test_progress_bar_with_various_speeds():
    """Test progress bar ETA with different iteration speeds."""
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        # Test 1: Fast iterations (short ETA)
        bar1 = ProgressBar(title='Fast', max_value=100, verbose_bool=True)
        time.sleep(0.05)
        bar1.update(50)  # 50% in 0.05s -> ETA should be very short
        output1 = sys.stdout.getvalue()
        assert '00:00:' in output1, "Fast progress should show short ETA"
        
        # Clear buffer
        sys.stdout = io.StringIO()
        
        # Test 2: Slow iterations (longer ETA)
        bar2 = ProgressBar(title='Slow', max_value=1000, verbose_bool=True)
        time.sleep(0.1)
        bar2.update(1)  # 0.1% in 0.1s -> ETA should be longer
        output2 = sys.stdout.getvalue()
        # With only 1 out of 1000 done in 0.1s, ETA will be ~99.9s = 00:01:39
        import re
        assert re.search(r'ETA: \d{2}:\d{2}:\d{2}', output2), "Should show ETA in hh:mm:ss format"
        
    finally:
        sys.stdout = old_stdout


def test_format_time_edge_cases():
    """Test edge cases for _format_time."""
    # Test very large time (more than 99 hours)
    assert _format_time(360000) == "100:00:00"  # 100 hours
    
    # Test negative time (should still format, though not expected in practice)
    # The function uses int() which truncates towards zero
    result = _format_time(-1)
    # -1 seconds: hours = -1 // 3600 = -1, minutes = (-1 % 3600) // 60, secs = -1 % 60
    # In Python: -1 % 60 = 59, (-1 % 3600) = 3599, 3599 // 60 = 59
    # So result should be -01:59:59
    assert ':' in result, "Should still format negative times"
