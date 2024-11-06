from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from torch_xla.experimental.pallas_kernels.multi_queries_paged_attention_kernel import paged_attention
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


# Set up paged_attention inputs.
def _generate_qkv(
    kv_seq_lens,
    page_size,
    max_kv_len,
    query_len,
    num_kv_heads,
    num_q_heads,
    head_dim,
    prng_key,
    dtype,
):
  assert max_kv_len % page_size == 0
  pages_per_sequence = max_kv_len // page_size
  batch_size = len(kv_seq_lens)
  total_pages = batch_size * pages_per_sequence
  k1, k2, k3, k4 = jax.random.split(prng_key, 4)
  k_pages = jax.random.normal(
      k1, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype)
  v_pages = jax.random.normal(
      k2, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype)

  page_indices = jnp.arange(batch_size * pages_per_sequence, dtype=jnp.int32)
  page_indices = jax.random.permutation(k3, page_indices, independent=True)
  page_indices = page_indices.reshape(batch_size, pages_per_sequence)
  q = jax.random.normal(
      k4, (batch_size, query_len, num_q_heads, head_dim), dtype=dtype)
  return q, k_pages, v_pages, page_indices


def _ref_jax_extended_paged_attention(
    q,  # [batch_size, query_len, num_query_heads, head_size]
    k_pages,  # [num_kv_heads, total_num_pages, page_size, head_size]
    v_pages,  # [num_kv_heads, total_num_pages, page_size, head_size]
    lengths,  # [batch_size], the effective kv_length.
    page_indices,  # [batch_size, pages_per_sequence]
    real_q_lens, # [batch_size] the effective q_length
):
  batch_size, query_len, num_query_heads, head_size = q.shape
  num_kv_heads, total_num_pages, page_size, _ = k_pages.shape
  num_query_per_kv = num_query_heads // num_kv_heads

  outputs = []
  for i in range(batch_size):
    kv_len = lengths[i]
    num_pages = (kv_len + page_size - 1) // page_size
    indices = page_indices[i, :num_pages]

    k = k_pages[:, indices]
    k = jnp.permute_dims(k, (1, 2, 0, 3))
    k = jnp.reshape(k, (num_pages * page_size, num_kv_heads, head_size))
    k = k[:kv_len]

    v = v_pages[:, indices]
    v = jnp.permute_dims(v, (1, 2, 0, 3))
    v = jnp.reshape(v, (num_pages * page_size, num_kv_heads, head_size))
    v = v[:kv_len]

    if num_query_per_kv != 1:
      k = jnp.repeat(k, num_query_per_kv, axis=1)
      v = jnp.repeat(v, num_query_per_kv, axis=1)

    attn = jnp.einsum("qhd,khd->hqk", q[i], k)
    attn = attn.astype('float32')
    real_q_len = real_q_lens[i]
    q_span = (kv_len - real_q_len) + jax.lax.broadcasted_iota(
        jnp.int32, (query_len, kv_len), 0)
    kv_span = jax.lax.broadcasted_iota(jnp.int32, (query_len, kv_len), 1)
    mask = jnp.where(q_span < kv_span, float("-inf"), 0.)
    with jax.numpy_rank_promotion("allow"):
      attn = attn + mask
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v)
    outputs.append(out)
  output = jnp.stack(outputs, axis=0)
  return output


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class PagedAttentionKernelTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()


#   def test_paged_attention(
#       self,
#   ):
#     dtype = jnp.bfloat16
#     page_size=16
#     num_kv_heads = 8
#     q_kv_head_ratio = 4
#     head_dim = 256
#     num_queries_per_compute_block = 32
#     block_kv_size = 256

#   @parameterized.product(
#       dtype=(jnp.float32, jnp.bfloat16),
#       page_size=(16, 32, 64),
#       num_kv_heads=(1, 8),
#       q_kv_head_ratio=(1, 4, 8),
#       head_dim=(128, 256),
#       num_queries_per_compute_block=(16, 32),
#       block_kv_size=(128, 256),
#   )
#   def test_paged_attention(
#       self,
#       dtype,
#       page_size,
#       num_kv_heads,
#       q_kv_head_ratio,
#       head_dim,
#       num_queries_per_compute_block,
#       block_kv_size,
#   ):
  def test_paged_attention(
      self,
  ):
    dtype = jnp.bfloat16
    page_size=16
    num_kv_heads = 8
    q_kv_head_ratio = 4
    head_dim = 256
    num_queries_per_compute_block = 32
    block_kv_size = 256

    max_kv_len = 2048
    query_len = 33
    kv_seq_lens = jax.random.randint(
        jax.random.key(0), (3,), 32, max_kv_len)

    assert query_len <= max_kv_len
    for cur_kv_seq in kv_seq_lens:
      assert query_len <= cur_kv_seq, f'{query_len} should be less than or equal to the kv_len {cur_kv_seq} in the current sequence.'
    batch_size = len(kv_seq_lens)
    pages_per_sequence = max_kv_len // page_size
    total_num_pages = batch_size * pages_per_sequence
    assert max_kv_len <= total_num_pages * page_size

    q, k_pages, v_pages, page_indices = _generate_qkv(
        kv_seq_lens,
        page_size,
        max_kv_len,
        query_len,
        num_kv_heads,
        num_kv_heads * q_kv_head_ratio,
        head_dim,
        jax.random.key(0),
        dtype,
    )

    print(f'Running paged_attention with {query_len=}')
    num_kv_pages_per_compute_block = block_kv_size // page_size
    actual_output = paged_attention(
        q,
        k_pages,
        v_pages,
        kv_seq_lens,
        page_indices,
        num_kv_pages_per_compute_block=num_kv_pages_per_compute_block,
        num_queries_per_compute_block=num_queries_per_compute_block,
    )
    # actual_output = jax.block_until_ready(actual_output)

    # Run the ref impl.
    expected_output = _ref_jax_extended_paged_attention(
        q,
        k_pages,
        v_pages,
        kv_seq_lens,
        page_indices,
    )

    self.assertEqual(actual_output.shape, expected_output.shape)

    if dtype == jnp.float32:
      atol = 1e-2
      rtol = 1e-2
    elif dtype == jnp.bfloat16:
      atol = 6e-1
      rtol = 1e-1
    else:
      self.fail(f'Unsupported dtype: {dtype}')
    self.assertTrue(
        jnp.allclose(expected_output, actual_output, atol=atol, rtol=rtol))

  def test_paged_attention_query_len_longer_than_kv_seq_len(
      self,
  ):
    dtype = jnp.float32
    page_size=16
    num_kv_heads = 8
    q_kv_head_ratio = 4
    head_dim = 256
    num_queries_per_compute_block = 32
    block_kv_size = 256

    max_kv_len = 2048
    # Set query_len(32)>kv_seq_lens(3)
    query_len = num_queries_per_compute_block
    real_query_len = jnp.array([2])
    kv_seq_lens = jnp.array([3])

    batch_size = len(kv_seq_lens)
    pages_per_sequence = max_kv_len // page_size
    total_num_pages = batch_size * pages_per_sequence
    assert max_kv_len <= total_num_pages * page_size

    q, k_pages, v_pages, page_indices = _generate_qkv(
        kv_seq_lens,
        page_size,
        max_kv_len,
        query_len,
        num_kv_heads,
        num_kv_heads * q_kv_head_ratio,
        head_dim,
        jax.random.key(0),
        dtype,
    )

    print(f'Running paged_attention with {query_len=}')
    num_kv_pages_per_compute_block = block_kv_size // page_size
    actual_output = paged_attention(
        q,
        k_pages,
        v_pages,
        kv_seq_lens,
        page_indices,
        num_kv_pages_per_compute_block=num_kv_pages_per_compute_block,
        num_queries_per_compute_block=num_queries_per_compute_block,
    )
    # actual_output = jax.block_until_ready(actual_output)

    # Run the ref impl.
    expected_output = _ref_jax_extended_paged_attention(
        q,
        k_pages,
        v_pages,
        kv_seq_lens,
        page_indices,
        real_query_len,
    )

    self.assertEqual(actual_output.shape, expected_output.shape)

    atol = 1e-2
    rtol = 1e-2
    print(f'Output max diff: {jnp.max(jnp.abs(expected_output - actual_output))}')
    print(f'Output mean diff: {jnp.mean(jnp.abs(expected_output - actual_output))}')
    self.assertTrue(
        jnp.allclose(expected_output, actual_output, atol=atol, rtol=rtol))

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
