# Uknee training runtime

`landmark` và `segment` có thể chạy từ repository hoặc được cài editable để
dùng ở bất kỳ working directory/notebook nào.

## Cài source một lần

Trong notebook, dùng đúng Python của kernel để tránh trường hợp `!python` trỏ
tới `/usr/bin/python` khác môi trường hiện tại:

```python
import sys
UKNEE_SOURCE = "/path/to/Uknee"
!{sys.executable} -m pip install --no-deps --no-build-isolation -e "{UKNEE_SOURCE}"
```

Sau đó kiểm tra:

```python
import sys
!{sys.executable} -c "import landmark, segment; print(landmark.__file__)"
!{sys.executable} -m landmark.train --help
```

`--project` là nơi chứa `data/` và `runs/`, không phải đường dẫn giúp Python
tìm source code. Có thể dùng project/data ở ổ khác với source repository.

Nếu không muốn cài package, chuyển working directory vào repository trước:

```python
%cd /path/to/Uknee
import sys
!{sys.executable} -m landmark.train --help
```

Xem CLI cụ thể tại `landmark/README.md` và `segment/README.md`.
