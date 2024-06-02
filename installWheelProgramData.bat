set UAI_SITE_=C:\ProgramData\uai\uai\Lib\site-packages
set UAI_PATH_=C:\ProgramData\uai\uai

call %UAI_PATH_%\Scripts\pip.exe install --target=%UAI_SITE_% dist\uaiDiffusers-1.1.5.7-py2.py3-none-any.whl --force-reinstall --upgrade