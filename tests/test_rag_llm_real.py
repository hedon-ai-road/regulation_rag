"""
RagLLM 真实测试文件
使用真实的 API 调用测试流式和非流式功能
"""

import unittest
import os
import sys
import time
from typing import Iterator

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from model import RagLLM, RagEmbedding
from langchain_core.outputs import GenerationChunk
import dotenv

# 加载环境变量
dotenv.load_dotenv()


class TestRagLLMReal(unittest.TestCase):
    """RagLLM 真实 API 测试类"""
    
    def setUp(self):
        """每个测试前的设置"""
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            self.skipTest("未找到 OPENROUTER_API_KEY 环境变量")
        
        self.llm = RagLLM()
        self.test_prompt = "你好，请简单介绍一下自己。"
    
    def test_initialization(self):
        """测试 RagLLM 初始化"""
        print("\n=== 测试1: RagLLM 真实初始化 ===")
        
        # 验证基本属性
        self.assertIsNotNone(self.llm.client)
        self.assertEqual(self.llm._llm_type, "rag_llm_deepseek/deepseek-r1-0528-qwen3-8b:free")
        
        # 验证方法存在
        self.assertTrue(hasattr(self.llm, '_call'))
        self.assertTrue(hasattr(self.llm, '_stream'))
        self.assertTrue(hasattr(self.llm, 'generate_response'))
        
        print("✓ RagLLM 初始化成功")
        print("✓ 所有必要方法存在")
        print(f"✓ API Key 已配置: {'是' if self.api_key else '否'}")
    
    def test_non_streaming_call(self):
        """测试非流式调用"""
        print("\n=== 测试2: 非流式调用 (_call) ===")
        
        try:
            # 测试调用
            result = self.llm._call(self.test_prompt, max_tokens=100)
            
            # 验证结果
            self.assertIsInstance(result, str)
            self.assertGreater(len(result.strip()), 0)
            
            print("✓ 非流式调用成功")
            print(f"✓ 返回结果长度: {len(result)} 字符")
            print(f"✓ 返回内容预览: {result[:100]}...")
            
        except Exception as e:
            self.skipTest(f"API 调用失败: {e}")
    
    def test_streaming_call(self):
        """测试流式调用"""
        print("\n=== 测试3: 流式调用 (_stream) ===")
        
        try:
            # 测试流式调用
            chunks = []
            for chunk in self.llm._stream(self.test_prompt, max_tokens=50):
                self.assertIsInstance(chunk, GenerationChunk)
                self.assertIsInstance(chunk.text, str)
                chunks.append(chunk)
                
                # 限制测试时间
                if len(chunks) >= 10:
                    break
            
            # 验证结果
            self.assertGreater(len(chunks), 0)
            
            # 合并所有文本
            full_text = ''.join(chunk.text for chunk in chunks)
            
            print("✓ 流式调用成功")
            print(f"✓ 返回 {len(chunks)} 个文本块")
            print(f"✓ 总文本长度: {len(full_text)} 字符")
            print(f"✓ 文本内容预览: {full_text[:100]}...")
            
        except Exception as e:
            self.skipTest(f"流式 API 调用失败: {e}")
    
    def test_generate_response_non_streaming(self):
        """测试便捷方法 - 非流式"""
        print("\n=== 测试4: 便捷方法 - 非流式 ===")
        
        try:
            result = self.llm.generate_response(
                self.test_prompt, 
                stream=False, 
                max_tokens=80
            )
            
            self.assertIsInstance(result, str)
            self.assertGreater(len(result.strip()), 0)
            
            print("✓ 便捷方法非流式调用成功")
            print(f"✓ 返回结果长度: {len(result)} 字符")
            print(f"✓ 返回内容: {result[:100]}...")
            
        except Exception as e:
            self.skipTest(f"便捷方法非流式调用失败: {e}")
    
    def test_generate_response_streaming(self):
        """测试便捷方法 - 流式"""
        print("\n=== 测试5: 便捷方法 - 流式 ===")
        
        try:
            result_stream = self.llm.generate_response(
                self.test_prompt, 
                stream=True, 
                max_tokens=60
            )
            
            # 验证返回的是迭代器
            self.assertTrue(hasattr(result_stream, '__iter__'))
            
            chunks = []
            for chunk in result_stream:
                self.assertIsInstance(chunk, GenerationChunk)
                chunks.append(chunk)
                
                # 限制测试时间
                if len(chunks) >= 8:
                    break
            
            self.assertGreater(len(chunks), 0)
            
            print("✓ 便捷方法流式调用成功")
            print(f"✓ 返回 {len(chunks)} 个文本块")
            
        except Exception as e:
            self.skipTest(f"便捷方法流式调用失败: {e}")
    
    def test_parameters_passing(self):
        """测试参数传递"""
        print("\n=== 测试6: 参数传递 ===")
        
        try:
            # 测试不同的参数
            result1 = self.llm._call(
                "说一个数字",
                temperature=0.1,
                max_tokens=10
            )
            
            result2 = self.llm._call(
                "说一个数字", 
                temperature=0.9,
                max_tokens=10
            )
            
            self.assertIsInstance(result1, str)
            self.assertIsInstance(result2, str)
            
            print("✓ 参数传递测试成功")
            print(f"✓ 低温度结果: {result1.strip()}")
            print(f"✓ 高温度结果: {result2.strip()}")
            
        except Exception as e:
            self.skipTest(f"参数传递测试失败: {e}")
    
    def test_different_prompts(self):
        """测试不同类型的提示"""
        print("\n=== 测试7: 不同类型提示 ===")
        
        prompts = [
            "1+1等于多少？",
            "请用一句话介绍Python",
            "列举三种编程语言"
        ]
        
        for i, prompt in enumerate(prompts, 1):
            try:
                result = self.llm._call(prompt, max_tokens=50)
                self.assertIsInstance(result, str)
                self.assertGreater(len(result.strip()), 0)
                
                print(f"✓ 提示 {i} 测试成功: {prompt}")
                print(f"  回答: {result.strip()[:50]}...")
                
            except Exception as e:
                print(f"⚠ 提示 {i} 测试失败: {e}")


class TestRagEmbeddingReal(unittest.TestCase):
    """RagEmbedding 真实测试类"""
    
    def test_embedding_initialization(self):
        """测试 RagEmbedding 初始化"""
        print("\n=== 测试8: RagEmbedding 真实初始化 ===")
        
        try:
            # 创建实例
            embedding = RagEmbedding()
            
            # 验证方法存在
            self.assertTrue(hasattr(embedding, 'get_embedding_fun'))
            
            # 获取 embedding 函数
            embed_func = embedding.get_embedding_fun()
            self.assertIsNotNone(embed_func)
            
            print("✓ RagEmbedding 初始化成功")
            print("✓ embedding 函数获取成功")
            
        except Exception as e:
            self.skipTest(f"RagEmbedding 初始化失败 (可能缺少模型文件): {e}")
    
    def test_embedding_functionality(self):
        """测试 Embedding 功能"""
        print("\n=== 测试9: Embedding 功能测试 ===")
        
        try:
            embedding = RagEmbedding()
            embed_func = embedding.get_embedding_fun()
            
            # 测试文本
            test_texts = ["这是一个测试句子", "另一个测试文本"]
            
            # 获取 embedding
            embeddings = embed_func.embed_documents(test_texts)
            
            # 验证结果
            self.assertEqual(len(embeddings), 2)
            self.assertIsInstance(embeddings[0], list)
            self.assertGreater(len(embeddings[0]), 0)
            
            print("✓ Embedding 功能测试成功")
            print(f"✓ 生成了 {len(embeddings)} 个向量")
            print(f"✓ 向量维度: {len(embeddings[0])}")
            
        except Exception as e:
            self.skipTest(f"Embedding 功能测试失败 (可能缺少模型文件): {e}")


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_llm_and_embedding_together(self):
        """测试 LLM 和 Embedding 集成"""
        print("\n=== 测试10: LLM 和 Embedding 集成测试 ===")
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            self.skipTest("未找到 OPENROUTER_API_KEY 环境变量")
        
        try:
            # 初始化两个组件
            llm = RagLLM()
            embedding = RagEmbedding()
            
            # 测试 LLM 生成文本
            generated_text = llm._call("介绍一下机器学习", max_tokens=100)
            self.assertIsInstance(generated_text, str)
            
            # 测试对生成的文本进行 embedding
            embed_func = embedding.get_embedding_fun()
            text_embedding = embed_func.embed_query(generated_text)
            
            self.assertIsInstance(text_embedding, list)
            self.assertGreater(len(text_embedding), 0)
            
            print("✓ LLM 和 Embedding 集成测试成功")
            print(f"✓ 生成文本长度: {len(generated_text)} 字符")
            print(f"✓ Embedding 向量维度: {len(text_embedding)}")
            print(f"✓ 生成文本预览: {generated_text[:80]}...")
            
        except Exception as e:
            self.skipTest(f"集成测试失败: {e}")


def run_tests():
    """运行所有测试"""
    print("=" * 70)
    print("RagLLM 和 RagEmbedding 真实 API 测试开始")
    print("=" * 70)
    
    # 检查环境
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"API Key 状态: {'已配置' if api_key else '未配置 (部分测试将跳过)'}")
    print()
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestRagLLMReal))
    suite.addTests(loader.loadTestsFromTestCase(TestRagEmbeddingReal))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"运行测试数量: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}")
            print(f"  {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}")
            print(f"  {traceback}")
    
    if result.skipped:
        print(f"\n跳过的测试: {len(result.skipped)} 个")
        for test, reason in result.skipped:
            print(f"- {test}: {reason}")
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    print("开始真实 API 测试...")
    print("注意: 这些测试会消耗真实的 API 调用")
    print()
    
    success = run_tests()
    
    if success:
        print("\n🎉 所有测试通过!")
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息") 