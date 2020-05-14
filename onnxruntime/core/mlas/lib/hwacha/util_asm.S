    .text
    .align 10
    .globl vtest_avi
vtest_avi: 
    vpset vp0
    vstop
    
    .globl vsetvlen
vsetvlen:
    vpset vp0
    vstop

    .globl vtest2
vtest2:
    vpset vp0
    vlb vv0, va1
    vlb vv1, va2 
    vlb vv2, va0
    vmul vv0, vv0, vv1
    vadd vv0, vv0, vv2
    vsb vv0, va0
    vstop



    