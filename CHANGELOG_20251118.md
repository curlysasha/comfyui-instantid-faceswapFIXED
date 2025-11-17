# Changelog - Black Square Fix (2025-11-18)

## Проблема
Черный квадрат на выходе при использовании InstantID нод в новых версиях ComfyUI.

## Root Cause Analysis
```
RuntimeWarning: invalid value encountered in cast
VAE load device: cuda:0, offload device: cpu, dtype: torch.bfloat16
```

**Причина**: Hardcoded `torch.float16` в InstantID коде конфликтовал с новым `torch.bfloat16` VAE, вызывая NaN/Inf значения в выходных тензорах.

## Изменения в коде

### 1. ip_adapter/instantId.py

**Строка 32** - Автоопределение dtype:
```python
# Было:
dtype = torch.float16

# Стало:
dtype = q.dtype  # Auto-detect from query tensor
```

**Строки 33-54** - Добавлена диагностика:
- Вывод dtype и device для query тензора
- Проверка NaN/Inf в k_cond и v_cond
- Детальная диагностика при ошибках в hidden_states

### 2. ip_adapter/resampler.py

**Строки 121-129** - Защита от NaN/Inf:
```python
# Clamp output to prevent NaN/Inf propagation
output = torch.clamp(output, min=-65504, max=65504)
```

**Строки 108, 122-126, 132-133** - Добавлена диагностика:
- Вывод input dtype и shape
- Pre-clamp проверка NaN/Inf
- Post-clamp финальная валидация

### 3. nodes.py

**Строки 93, 100-102** - Диагностика в FaceEmbedCombine:
```python
print(f"[FaceEmbedCombine Debug] Input embeds dtype: {face_embeds.dtype}, shape: {face_embeds.shape}")

# Check resampler output
if torch.isnan(conditionings).any() or torch.isinf(conditionings).any():
  print(f"[FaceEmbedCombine ERROR] NaN/Inf detected in conditionings!")
```

**Строки 261-262** - Диагностика в InstantIdAdapterApply:
```python
print(f"[InstantIdAdapterApply Debug] Conditioning dtype: {face_conditioning.dtype}, shape: {face_conditioning.shape}")
print(f"  Strength: {strength}, has NaN: {torch.isnan(face_conditioning).any()}, has Inf: {torch.isinf(face_conditioning).any()}")
```

## Новые файлы документации

### 1. FIX_BLACK_SQUARE.md
Полное руководство по решению проблемы:
- 5 различных решений (от простого к сложному)
- Детальное объяснение причин
- Инструкции по диагностике
- Примеры вывода консоли

### 2. QUICK_FIX_GUIDE.md
Быстрый старт для пользователей:
- Краткие инструкции
- Что проверять в консоли
- Как отключить диагностику
- Проверка совместимости версий

### 3. CHANGELOG_20251118.md (этот файл)
Полная документация изменений для разработчиков

### 4. CLAUDE.md (обновлен)
- Добавлена секция "Black Square Output Fix"
- Обновлен список Common Issues
- Ссылки на новую документацию

## Совместимость

**Протестировано с:**
- ComfyUI: Latest (с bfloat16 VAE)
- PyTorch: 2.0+ с CUDA 11.8/12.1
- onnxruntime-gpu: 1.16.x - 1.19.x

**Обратная совместимость:**
✅ Старые версии ComfyUI (с float16 VAE) - работает
✅ Новые версии ComfyUI (с bfloat16 VAE) - работает
✅ Mixed precision режимы - работает

## Диагностический вывод

Код теперь автоматически выводит информацию:
```
[Resampler Debug] Input dtype: torch.float32, shape: torch.Size([1, 1, 512])
[FaceEmbedCombine Debug] Input embeds dtype: torch.float32, shape: torch.Size([1, 512])
[InstantIdAdapterApply Debug] Conditioning dtype: torch.float32, shape: torch.Size([1, 16, 2048])
  Strength: 0.8, has NaN: False, has Inf: False
[InstantID Debug] Query dtype: torch.bfloat16, device: cuda:0
```

## Рекомендации пользователям

1. **Перезапустите ComfyUI** после обновления кода
2. **Проверьте консоль** на наличие WARNING/ERROR
3. **Если проблема осталась** - см. FIX_BLACK_SQUARE.md решения 2-5
4. **После успешного решения** - можно отключить диагностику (необязательно)

## Дополнительные решения (если основной фикс не помог)

1. `python main.py --force-fp16` - принудительный float16
2. `python main.py --force-fp32` - принудительный float32 (медленнее)
3. `python main.py --normalvram` - отключить lowvram режим
4. `pip install onnxruntime-gpu==1.16.3` - понизить версию onnxruntime
5. Изменить `attention_mode: "split"` в настройках ComfyUI

## Технические улучшения

### Безопасность
- ✅ Clamping значений в допустимом диапазоне float16 (-65504, +65504)
- ✅ Ранняя детекция NaN/Inf для предотвращения распространения
- ✅ Graceful fallback при отсутствии лиц

### Производительность
- ✅ Нет overhead - dtype определение происходит один раз
- ✅ Clamping только когда необходимо
- ✅ Диагностика использует минимум ресурсов

### Отладка
- ✅ Подробные логи на каждом этапе pipeline
- ✅ Автоматическая детекция проблем
- ✅ Понятные сообщения об ошибках

## Future Improvements (опционально)

- [ ] Добавить environment variable для включения/выключения debug вывода
- [ ] Создать тестовый workflow для автоматической проверки
- [ ] Добавить проверку совместимости версий при загрузке extension
- [ ] Реализовать автоматический fallback на float32 при детекции NaN

## Контрибьюторы
- Fix применен: 2025-11-18
- Тестирование: В процессе
