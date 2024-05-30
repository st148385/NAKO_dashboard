### Integrate Transformations
First one has to implement the transformation inside of transforms.
Check the file for examples.

Secondly the transforms need to be added in TRANSFORMS and TRANSFORMS_DICT inside of enum.

After that the transformations can be used on specific column if defined in the gin config:
```python
Metadata30kWorkflow.feature_selection = {
    '0': %TRANSFORMS.DROP,
    '0_right': %TRANSFORMS.DROP,
    '1': %TRANSFORMS.DROP,
}
```