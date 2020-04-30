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
    vlb vv0, va0
    vadd vv0, vv0, vv0
    vsb vv0, va1

    vstop



    