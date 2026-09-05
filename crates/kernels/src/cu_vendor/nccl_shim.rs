//! Thin `extern "C"` bindings to NCCL's real C API (`/usr/include/nccl.h`,
//! NCCL 2.31.2 as installed on `bw` via `apt install libnccl2 libnccl-dev`).
//! Nothing here is compiled by us — NCCL ships its own prebuilt shared
//! library (`libnccl.so`), so this is bindings-only, unlike the AOT-compiled
//! CUTLASS/FA2 shims that live alongside this file.
//!
//! Every constant and struct layout below was checked against the real
//! installed header rather than assumed:
//! - `NCCL_UNIQUE_ID_BYTES = 128` (`nccl.h:40`)
//! - `ncclSum = 0` (`nccl.h:448`)
//! - `ncclFloat32 = 7` (`nccl.h:473`)
//! - `ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)`
//!   and `ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm,
//!   stream)` argument order match `nccl.h`'s real declarations exactly.

#![allow(non_camel_case_types)] // mirrors NCCL's own C identifier casing on purpose

use std::ffi::c_void;

pub const NCCL_UNIQUE_ID_BYTES: usize = 128;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ncclUniqueId {
    pub internal: [u8; NCCL_UNIQUE_ID_BYTES],
}

pub type ncclComm_t = *mut c_void;

#[repr(i32)]
#[derive(Clone, Copy)]
pub enum ncclDataType_t {
    ncclFloat32 = 7,
    ncclUint8 = 1,
}

#[repr(i32)]
#[derive(Clone, Copy)]
pub enum ncclRedOp_t {
    ncclSum = 0,
}

unsafe extern "C" {
    pub fn ncclGetUniqueId(unique_id: *mut ncclUniqueId) -> i32;
    pub fn ncclCommInitRank(
        comm: *mut ncclComm_t,
        nranks: i32,
        comm_id: ncclUniqueId,
        rank: i32,
    ) -> i32;
    pub fn ncclCommDestroy(comm: ncclComm_t) -> i32;
    pub fn ncclAllReduce(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: cudarc::driver::sys::CUstream,
    ) -> i32;
    pub fn ncclBroadcast(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: i32,
        comm: ncclComm_t,
        stream: cudarc::driver::sys::CUstream,
    ) -> i32;
}
