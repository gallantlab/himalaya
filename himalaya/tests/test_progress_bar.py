from himalaya.progress_bar import bar
from himalaya.progress_bar import ProgressBar


def test_progress_bar():
    # simple smoke test
    for ii in bar(range(10)):
        pass

    bar_ = ProgressBar(title="La barre", max_value=10, initial_value=0,
                       max_chars=40, progress_character='.', spinner=False,
                       verbose_bool=True)
    for ii in bar_(range(10)):
        pass

    bar_ = ProgressBar(max_value=10)
    for ii in range(10):
        bar_.update_with_increment_value(1)
    bar_.close()
