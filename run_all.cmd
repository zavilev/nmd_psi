@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" py310

echo ==========================================
echo Running: all_nmd_inactivation_data.ipynb
echo ==========================================
jupyter nbconvert --to notebook --execute "all_nmd_inactivation_data.ipynb" --inplace --ExecutePreprocessor.kernel_name=python3
if errorlevel 1 goto :error

echo ==========================================
echo Running: psi_and_expression_gtex.ipynb
echo ==========================================
jupyter nbconvert --to notebook --execute "psi_and_expression_gtex.ipynb" --inplace --ExecutePreprocessor.kernel_name=python3
if errorlevel 1 goto :error

echo ==========================================
echo Running: Figures.ipynb
echo ==========================================
jupyter nbconvert --to notebook --execute "Figures.ipynb" --inplace --ExecutePreprocessor.kernel_name=python3
if errorlevel 1 goto :error

echo ==========================================
echo Running: encode_qc.ipynb
echo ==========================================
jupyter nbconvert --to notebook --execute "encode_qc.ipynb" --inplace --ExecutePreprocessor.kernel_name=python3
if errorlevel 1 goto :error

echo ==========================================
echo Running: pdflatex figures.tex
echo ==========================================
pdflatex figures.tex
if errorlevel 1 goto :error

echo.
echo ==========================================
echo Everything completed successfully.
echo ==========================================
goto :end

:error
echo.
echo ==========================================
echo ERROR: Execution failed.
echo ==========================================

:end