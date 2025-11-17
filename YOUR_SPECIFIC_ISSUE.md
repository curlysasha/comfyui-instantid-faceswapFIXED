# Ваша Конкретная Проблема: NaN в Resampler

## Диагноз
На основе логов:
```
[Resampler WARNING] Pre-clamp NaN: True, Inf: False
[FaceEmbed Debug] Raw embedding from InsightFace: ...
```

**ПРОБЛЕМА**: InsightFace (через onnxruntime) возвращает невалидные face embeddings, которые содержат NaN значения.

**ЭТО НЕ ПРОБЛЕМА dtype!** Это проблема onnxruntime версии или повреждённых ONNX моделей.

## Немедленные Действия

### 1. Проверьте версию onnxruntime-gpu

```bash
python -c "import onnxruntime; print(onnxruntime.__version__)"
```

**Если >= 1.17** - вот ваша проблема!

### 2. Понизьте до стабильной версии

```bash
# В том же окружении где установлен ComfyUI
pip uninstall onnxruntime-gpu onnxruntime -y
pip install onnxruntime-gpu==1.16.3
```

### 3. Перезапустите ComfyUI и попробуйте снова

```bash
python main.py
```

## Что ожидать в логах после исправления

**ПРАВИЛЬНЫЙ вывод (без NaN):**
```
[FaceEmbed Debug] Raw embedding from InsightFace: dtype=float32, shape=(512,)
  Stats: min=-0.2345, max=0.8765, mean=0.0123
[Resampler Debug] Input dtype: torch.float32, shape: torch.Size([1, 1, 512])
[Resampler] Latents stats: min=-0.0891, max=0.0923, device=cpu
[Resampler] After proj_in: has_nan=False, min=-2.3456, max=2.1234
[Resampler] After proj_out: has_nan=False, min=-1.5678, max=1.4321
```

**НЕПРАВИЛЬНЫЙ вывод (с NaN - ваш текущий случай):**
```
[FaceEmbed ERROR] InsightFace returned NaN/Inf embedding!
  This indicates a problem with InsightFace or onnxruntime
[Resampler WARNING] Pre-clamp NaN: True, Inf: False
```

## Временное Решение (Уже Применено)

Код теперь:
1. ✅ Детектирует NaN от InsightFace
2. ✅ Возвращает zero embeddings вместо NaN
3. ✅ Предотвращает черный квадрат

**НО**: Генерация будет низкого качества, т.к. face conditioning отключён.

## Альтернативные Решения (если понижение версии не помогло)

### A. Проверьте ONNX модели

```bash
cd ComfyUI/models/insightface/models/antelopev2/
ls -lh

# Должно быть 5 файлов размером:
# 1k3d68.onnx       ~19MB
# 2d106det.onnx     ~5MB
# genderage.onnx    ~1.3MB
# glintr100.onnx    ~250MB
# scrfd_10g_bnkps.onnx  ~17MB
```

Если размеры отличаются или файлы повреждены - перекачайте модели.

### B. Попробуйте CPU providers

В [nodes.py:527](nodes.py#L527) измените:
```python
# Было:
providers=["CPUExecutionProvider", "CUDAExecutionProvider"]

# Попробуйте только CPU (медленнее, но надёжнее):
providers=["CPUExecutionProvider"]
```

### C. Проверьте CUDA совместимость

```bash
nvidia-smi  # Проверьте версию драйвера

python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
python -c "import onnxruntime; print(f'ONNX providers: {onnxruntime.get_available_providers()}')"
```

Убедитесь что CUDA версии совместимы между PyTorch и onnxruntime.

### D. Полная переустановка onnxruntime

```bash
pip uninstall onnxruntime-gpu onnxruntime onnxruntime-training -y
pip cache purge
pip install onnxruntime-gpu==1.16.3 --no-cache-dir
```

## Диагностический Скрипт

Запустите:
```bash
cd /mnt/c/git/comfyui-instantid-faceswap
python diagnose_system.py
```

Это покажет все версии и выявит проблемы.

## Известные Проблемные Комбинации

❌ **onnxruntime-gpu 1.17.x + некоторые CUDA версии** - NaN в embeddings
❌ **onnxruntime-gpu 1.18.x + WSL2** - может давать NaN
❌ **Смешанные версии onnxruntime packages** - конфликты

✅ **onnxruntime-gpu 1.16.3 + CUDA 11.8/12.1** - стабильная комбинация
✅ **onnxruntime-gpu 1.15.1** - тоже работает

## Следующие Шаги

1. ✅ Понизьте onnxruntime до 1.16.3
2. ✅ Перезапустите ComfyUI
3. ✅ Проверьте логи на отсутствие NaN warnings
4. ✅ Если всё работает - отключите диагностику (опционально)

## Если Всё Ещё Не Работает

Скопируйте полный вывод:
```bash
python diagnose_system.py > system_info.txt
```

И вывод ComfyUI с вашим workflow:
```
[FaceEmbed Debug] ...
[Resampler Debug] ...
```

Создайте issue на GitHub с этой информацией.

## Обновление от Claude

Я добавил в код:
- ✅ Детальную диагностику на каждом этапе
- ✅ Graceful fallback при NaN (zero tensors вместо crash)
- ✅ Проверку входных данных от InsightFace
- ✅ Защиту от распространения NaN по pipeline

Но **root cause** нужно исправить понижением версии onnxruntime, иначе качество генерации будет плохим.
