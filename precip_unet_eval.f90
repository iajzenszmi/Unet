
program precip_unet_eval
  implicit none
  integer :: narg
  character(len=256) :: f_obs, f_mrms, f_imerg, f_mw
  real(8) :: thresh
  integer :: perday, permonth
  integer :: i, n, ios
  real(8), allocatable :: obs(:), pm(:), pi(:), pw(:)

  thresh   = 0.1d0
  perday   = 24
  permonth = 720

  call parse_args(f_obs, f_mrms, f_imerg, f_mw, thresh, perday, permonth)
  call read_file(f_obs,   obs)
  call read_file(f_mrms,  pm)
  call read_file(f_imerg, pi)
  call read_file(f_mw,    pw)

  n = size(obs)
  if (size(pm) /= n .or. size(pi) /= n .or. size(pw) /= n) then
     write(*,*) 'ERROR: Input files have different lengths.'
     stop 1
  end if

  call banner('NATIVE (no aggregation)')
  call report_block(obs, pm, 'MRMS ', thresh)
  call report_block(obs, pi, 'IMERG', thresh)
  call report_block(obs, pw, 'MWCOM', thresh)

  if (perday > 0 .and. n >= perday) then
     call banner('DAILY aggregation')
     call aggregate_and_report(obs, pm, 'MRMS ', thresh, perday)
     call aggregate_and_report(obs, pi, 'IMERG', thresh, perday)
     call aggregate_and_report(obs, pw, 'MWCOM', thresh, perday)
  end if

  if (permonth > 0 .and. n >= permonth) then
     call banner('MONTHLY aggregation')
     call aggregate_and_report(obs, pm, 'MRMS ', thresh, permonth)
     call aggregate_and_report(obs, pi, 'IMERG', thresh, permonth)
     call aggregate_and_report(obs, pw, 'MWCOM', thresh, permonth)
  end if

contains

  subroutine parse_args(f1,f2,f3,f4,th, pd, pmnth)
    character(len=*), intent(out) :: f1,f2,f3,f4
    real(8), intent(inout) :: th
    integer, intent(inout) :: pd, pmnth
    integer :: i, argc
    character(len=256) :: arg

    argc = command_argument_count()
    if (argc < 4) then
       call usage()
       stop 2
    end if
    call get_command_argument(1, f1)
    call get_command_argument(2, f2)
    call get_command_argument(3, f3)
    call get_command_argument(4, f4)

    i = 5
    do while (i <= argc)
      call get_command_argument(i, arg)
      if (trim(arg) == '--thresh') then
         if (i+1 <= argc) then
            call get_command_argument(i+1, arg); read(arg,*) th; i = i + 2; cycle
         else
            call usage(); stop 3
         end if
      else if (trim(arg) == '--perday') then
         if (i+1 <= argc) then
            call get_command_argument(i+1, arg); read(arg,*) pd; i = i + 2; cycle
         else
            call usage(); stop 3
         end if
      else if (trim(arg) == '--permonth') then
         if (i+1 <= argc) then
            call get_command_argument(i+1, arg); read(arg,*) pmnth; i = i + 2; cycle
         else
            call usage(); stop 3
         end if
      else
         write(*,*) 'Unknown option: ', trim(arg)
         call usage(); stop 3
      end if
    end do
  end subroutine parse_args

  subroutine usage()
    write(*,*) 'Usage: eval stage4.txt mrms.txt imerg.txt mwcomb.txt [--thresh x] [--perday n] [--permonth m]'
    write(*,*) 'Example: ./eval stage4.txt mrms.txt imerg.txt mwcomb.txt --thresh 0.1 --perday 24 --permonth 720'
  end subroutine usage

  subroutine read_file(fname, arr)
    character(len=*), intent(in) :: fname
    real(8), allocatable, intent(out) :: arr(:)
    integer :: iostat, nlines
    real(8) :: tmp
    character(len=4096) :: line
    integer :: unit

    open(newunit=unit, file=trim(fname), status='old', action='read', iostat=iostat)
    if (iostat /= 0) then
       write(*,*) 'ERROR: Cannot open ', trim(fname)
       stop 4
    end if

    nlines = 0
    do
      read(unit, '(A)', iostat=iostat) line
      if (iostat /= 0) exit
      if (len_trim(line) > 0) nlines = nlines + 1
    end do
    if (nlines <= 0) then
      write(*,*) 'ERROR: Empty file ', trim(fname)
      stop 5
    end if
    rewind(unit)

    allocate(arr(nlines))
    nlines = 0
    do
      read(unit, *, iostat=iostat) tmp
      if (iostat /= 0) exit
      nlines = nlines + 1
      arr(nlines) = tmp
    end do
    close(unit)
  end subroutine read_file

  subroutine banner(title)
    character(len=*), intent(in) :: title
    write(*,*) ''
    write(*,*) '==================== ', trim(title), ' ===================='
    write(*,*) 'Model  |   CC    POD     FAR     CSI     Bias'
    write(*,*) '-------+-----------------------------------------'
  end subroutine banner

  subroutine report_block(obs, pred, name, th)
    real(8), intent(in) :: obs(:), pred(:), th
    character(len=*), intent(in) :: name
    real(8) :: cc, pod, far, csi, bias
    call metrics(obs, pred, th, cc, pod, far, csi, bias)
    write(*,'(A6,1X,"|",1X,F6.3,1X,F6.3,1X,F7.3,1X,F7.3,1X,F7.3)') name, cc, pod, far, csi, bias
  end subroutine report_block

  subroutine aggregate_and_report(obs, pred, name, th, bucket)
    real(8), intent(in) :: obs(:), pred(:), th
    integer, intent(in) :: bucket
    character(len=*), intent(in) :: name
    real(8), allocatable :: o2(:), p2(:)
    integer :: nfull, i, k, j, base

    nfull = (size(obs) / bucket)
    if (nfull <= 0) then
       write(*,*) 'Insufficient samples for bucket size ', bucket
       return
    end if
    allocate(o2(nfull), p2(nfull))
    do i = 1, nfull
      base = (i-1)*bucket
      o2(i) = 0.0d0
      p2(i) = 0.0d0
      do j = 1, bucket
        o2(i) = o2(i) + obs(base+j)
        p2(i) = p2(i) + pred(base+j)
      end do
      o2(i) = o2(i) / real(bucket,8)
      p2(i) = p2(i) / real(bucket,8)
    end do
    call report_block(o2, p2, name, th)
    deallocate(o2, p2)
  end subroutine aggregate_and_report

  subroutine metrics(obs, pred, th, cc, pod, far, csi, bias)
    real(8), intent(in) :: obs(:), pred(:), th
    real(8), intent(out) :: cc, pod, far, csi, bias
    integer :: n, i
    real(8) :: mo, mp, so, sp, cov, dobs, dpred
    integer :: hits, misses, falsea, correctn
    n = size(obs)

    mo = 0d0; mp = 0d0
    do i = 1, n
      mo = mo + obs(i)
      mp = mp + pred(i)
    end do
    mo = mo / real(n,8)
    mp = mp / real(n,8)

    so = 0d0; sp = 0d0; cov = 0d0
    do i = 1, n
      dobs  = obs(i)  - mo
      dpred = pred(i) - mp
      so = so + dobs*dobs
      sp = sp + dpred*dpred
      cov = cov + dobs*dpred
    end do
    if (so > 0d0 .and. sp > 0d0) then
      cc = cov / sqrt(so*sp)
    else
      cc = 0d0
    end if

    hits = 0; misses = 0; falsea = 0; correctn = 0
    do i = 1, n
      if (obs(i) >= th) then
        if (pred(i) >= th) then
          hits = hits + 1
        else
          misses = misses + 1
        end if
      else
        if (pred(i) >= th) then
          falsea = falsea + 1
        else
          correctn = correctn + 1
        end if
      end if
    end do

    if (hits + misses > 0) then
      pod = real(hits,8) / real(hits + misses,8)
    else
      pod = 0d0
    end if
    if (hits + falsea > 0) then
      far = real(falsea,8) / real(hits + falsea,8)
    else
      far = 0d0
    end if
    if (hits + misses + falsea > 0) then
      csi = real(hits,8) / real(hits + misses + falsea,8)
    else
      csi = 0d0
    end if

    if (sum(obs) > 0d0) then
      bias = sum(pred) / sum(obs)
    else
      bias = 0d0
    end if
  end subroutine metrics

end program precip_unet_eval
