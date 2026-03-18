
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/compute.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef S
bool LUdecompose_s(ALPHA_INT row, ALPHA_INT col, float *x, float *y)
{
    bool flag = true;    
    float high = 0.0f, low = 0.0f, total = 0.0f;
    float yy[row];

    for(ALPHA_INT iter_r = 1; iter_r < row; iter_r++)
        x[iter_r * col] = x[iter_r *col]/x[0];

    for(ALPHA_INT iter_r = 1; iter_r < row; iter_r++)
    {
        for(ALPHA_INT inner_r = iter_r; inner_r <row; inner_r++)
        {
            high = 0.f;
            for(ALPHA_INT sec_inner_r = 0; sec_inner_r < iter_r; sec_inner_r++)
                high += x[iter_r * col + sec_inner_r] * x[sec_inner_r * col + inner_r];
            x[iter_r * col + inner_r] = x[iter_r * col + inner_r] - high;
        }

        for(ALPHA_INT inner_r = iter_r + 1; inner_r < col && iter_r != col - 1; inner_r++)
        {
            low = 0.f;
            for(ALPHA_INT sec_inner_r = 0; sec_inner_r < iter_r; sec_inner_r++)
                low += x[inner_r * col + sec_inner_r] * x[sec_inner_r * col + iter_r];
            x[inner_r * col + iter_r] = (x[inner_r * col + iter_r] - low) / x[iter_r * col + iter_r];
        }
    }

    for(ALPHA_INT iter_r = 0; iter_r < row; iter_r++)
        if(x[iter_r * col + iter_r]==0) return false;

    yy[0] = y[0];
    
    for(ALPHA_INT iter_r = 1; iter_r < row; iter_r++)
    {
        total = 0.0f;
        for(ALPHA_INT inner_r = 0; inner_r < iter_r; inner_r++)
            total += x[iter_r * col + inner_r] * yy[inner_r];
        yy[iter_r] = y[iter_r] - total;
    }

    y[row - 1] = yy[row-1] / x[(row - 1) * col + col-1];
    for(ALPHA_INT iter_r = row - 2; iter_r >=0 ; iter_r--)
    {
        total = 0.0f;
        for(ALPHA_INT inner_r = iter_r + 1; inner_r < row; inner_r++)
            total += x[iter_r * col + inner_r] * y[inner_r];
        y[iter_r] = (yy[iter_r] - total) / x[iter_r * col + iter_r];
    }
    return flag;
}
#elif defined(D)
bool LUdecompose_d(ALPHA_INT row, ALPHA_INT col, double *x, double *y)
{
    bool flag = true;    
    double high = 0.0f, low = 0.0f, total = 0.0f;
    double yy[row];

    for(ALPHA_INT iter_r = 1; iter_r < row; iter_r++)
        x[iter_r * col] = x[iter_r *col]/x[0];

    for(ALPHA_INT iter_r = 1; iter_r < row; iter_r++)
    {
        for(ALPHA_INT inner_r = iter_r; inner_r <row; inner_r++)
        {
            high = 0.f;
            for(ALPHA_INT sec_inner_r = 0; sec_inner_r < iter_r; sec_inner_r++)
                high += x[iter_r * col + sec_inner_r] * x[sec_inner_r * col + inner_r];
            x[iter_r * col + inner_r] = x[iter_r * col + inner_r] - high;
        }

        for(ALPHA_INT inner_r = iter_r + 1; inner_r < col && iter_r != col - 1; inner_r++)
        {
            low = 0.f;
            for(ALPHA_INT sec_inner_r = 0; sec_inner_r < iter_r; sec_inner_r++)
                low += x[inner_r * col + sec_inner_r] * x[sec_inner_r * col + iter_r];
            x[inner_r * col + iter_r] = (x[inner_r * col + iter_r] - low) / x[iter_r * col + iter_r];
        }
    }

    for(ALPHA_INT iter_r = 0; iter_r < row; iter_r++)
        if(x[iter_r * col + iter_r]==0) return false;

    yy[0] = y[0];
    
    for(ALPHA_INT iter_r = 1; iter_r < row; iter_r++)
    {
        total = 0.0f;
        for(ALPHA_INT inner_r = 0; inner_r < iter_r; inner_r++)
            total += x[iter_r * col + inner_r] * yy[inner_r];
        yy[iter_r] = y[iter_r] - total;
    }

    y[row - 1] = yy[row-1] / x[(row - 1) * col + col-1];
    for(ALPHA_INT iter_r = row - 2; iter_r >=0 ; iter_r--)
    {
        total = 0.0f;
        for(ALPHA_INT inner_r = iter_r + 1; inner_r < row; inner_r++)
            total += x[iter_r * col + inner_r] * y[inner_r];
        y[iter_r] = (yy[iter_r] - total) / x[iter_r * col + iter_r];
    }
    return flag;
}
#elif defined(C)
bool LUdecompose_c(ALPHA_INT row, ALPHA_INT col, ALPHA_Complex8 *x, ALPHA_Complex8 *y)
{
    bool flag = true;    
    ALPHA_Complex8 high = {0.0f, 0.0f}, low = {0.0f, 0.0f}, total = {0.0f, 0.0f};
    ALPHA_Complex8 yy[row];

    for(ALPHA_INT iter_r = 1; iter_r < row; iter_r++)
        cmp_div(x[iter_r * col], x[iter_r *col], x[0]);

    for(ALPHA_INT iter_r = 1; iter_r < row; iter_r++)
    {
        for(ALPHA_INT inner_r = iter_r; inner_r <row; inner_r++)
        {
            high.real = 0.0f;
            high.imag = 0.0f;

            for(ALPHA_INT sec_inner_r = 0; sec_inner_r < iter_r; sec_inner_r++)
                cmp_madde(high, x[iter_r * col + sec_inner_r], x[sec_inner_r * col + inner_r]);
            
            cmp_sub(x[iter_r * col + inner_r], x[iter_r * col + inner_r], high);
        }

        for(ALPHA_INT inner_r = iter_r + 1; inner_r < col && iter_r != col - 1; inner_r++)
        {
            low.real = 0.0f;
            low.imag = 0.0f;

            for(ALPHA_INT sec_inner_r = 0; sec_inner_r < iter_r; sec_inner_r++)
                cmp_madde(low, x[inner_r * col + sec_inner_r], x[sec_inner_r * col + iter_r]);
            
            cmp_sub(x[inner_r * col + iter_r], x[inner_r * col + iter_r], low);
            cmp_div(x[inner_r * col + iter_r], x[inner_r * col + iter_r], x[iter_r * col + iter_r]);
        }
    }

    for(ALPHA_INT iter_r = 0; iter_r < row; iter_r++)
        if(x[iter_r * col + iter_r].real == 0.0f && x[iter_r * col + iter_r].imag == 0.0f) return false;

    yy[0] = y[0];
    
    for(ALPHA_INT iter_r = 1; iter_r < row; iter_r++)
    {
        total.real = 0.0f;
        total.imag = 0.0f;

        for(ALPHA_INT inner_r = 0; inner_r < iter_r; inner_r++)
            cmp_madde(total, x[iter_r * col + inner_r], yy[inner_r]);

        cmp_sub(yy[iter_r], y[iter_r], total);
    }

    cmp_div(y[row - 1], yy[row-1], x[(row - 1) * col + col-1]);

    for(ALPHA_INT iter_r = row - 2; iter_r >=0 ; iter_r--)
    {
        total.real = 0.0f;
        total.imag = 0.0f;

        for(ALPHA_INT inner_r = iter_r + 1; inner_r < row; inner_r++)
            cmp_madde(total, x[iter_r * col + inner_r], y[inner_r]);

        cmp_sub(y[iter_r], yy[iter_r], total);    
        cmp_div(y[iter_r], y[iter_r], x[iter_r * col + iter_r]);
    }
    return flag;
}
#else
bool LUdecompose_z(ALPHA_INT row, ALPHA_INT col, ALPHA_Complex16 *x, ALPHA_Complex16 *y)
{
    bool flag = true;    
    ALPHA_Complex16 high = {0.0f, 0.0f}, low = {0.0f, 0.0f}, total = {0.0f, 0.0f};
    ALPHA_Complex16 yy[row];

    for(ALPHA_INT iter_r = 1; iter_r < row; iter_r++)
        cmp_div(x[iter_r * col], x[iter_r *col], x[0]);

    for(ALPHA_INT iter_r = 1; iter_r < row; iter_r++)
    {
        for(ALPHA_INT inner_r = iter_r; inner_r <row; inner_r++)
        {
            high.real = 0.0f;
            high.imag = 0.0f;

            for(ALPHA_INT sec_inner_r = 0; sec_inner_r < iter_r; sec_inner_r++)
                cmp_madde(high, x[iter_r * col + sec_inner_r], x[sec_inner_r * col + inner_r]);
            
            cmp_sub(x[iter_r * col + inner_r], x[iter_r * col + inner_r], high);
        }

        for(ALPHA_INT inner_r = iter_r + 1; inner_r < col && iter_r != col - 1; inner_r++)
        {
            low.real = 0.0f;
            low.imag = 0.0f;

            for(ALPHA_INT sec_inner_r = 0; sec_inner_r < iter_r; sec_inner_r++)
                cmp_madde(low, x[inner_r * col + sec_inner_r], x[sec_inner_r * col + iter_r]);
            
            cmp_sub(x[inner_r * col + iter_r], x[inner_r * col + iter_r], low);
            cmp_div(x[inner_r * col + iter_r], x[inner_r * col + iter_r], x[iter_r * col + iter_r]);
        }
    }

    for(ALPHA_INT iter_r = 0; iter_r < row; iter_r++)
        if(x[iter_r * col + iter_r].real == 0.0f && x[iter_r * col + iter_r].imag == 0.0f) return false;

    yy[0] = y[0];
    
    for(ALPHA_INT iter_r = 1; iter_r < row; iter_r++)
    {
        total.real = 0.0f;
        total.imag = 0.0f;

        for(ALPHA_INT inner_r = 0; inner_r < iter_r; inner_r++)
            cmp_madde(total, x[iter_r * col + inner_r], yy[inner_r]);

        cmp_sub(yy[iter_r], y[iter_r], total);
    }

    cmp_div(y[row - 1], yy[row-1], x[(row - 1) * col + col-1]);

    for(ALPHA_INT iter_r = row - 2; iter_r >=0 ; iter_r--)
    {
        total.real = 0.0f;
        total.imag = 0.0f;

        for(ALPHA_INT inner_r = iter_r + 1; inner_r < row; inner_r++)
            cmp_madde(total, x[iter_r * col + inner_r], y[inner_r]);

        cmp_sub(y[iter_r], yy[iter_r], total);    
        cmp_div(y[iter_r], y[iter_r], x[iter_r * col + iter_r]);
    }
    return flag;
}
#endif