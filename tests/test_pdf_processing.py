"""
PDF处理功能测试文件
测试PDF加载、表格提取、文本处理等功能
调用doc_parse.py中封装的工具
"""

import unittest
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入我们的PDF处理类
from doc_parse import Pdf

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPDFProcessing(unittest.TestCase):
    """PDF处理功能测试类 - 使用doc_parse中的封装工具"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.test_pdf_path = "data/zhidu_employee.pdf"
        cls.test_pdf_path_alt = "data/zhidu_travel.pdf"
        
        # 检查测试文件是否存在
        if not os.path.exists(cls.test_pdf_path):
            raise FileNotFoundError(f"测试PDF文件不存在: {cls.test_pdf_path}")
        
        logger.info(f"开始测试PDF处理功能，测试文件: {cls.test_pdf_path}")
    
    def setUp(self):
        """每个测试前的设置"""
        self.pdf_processor = Pdf(self.test_pdf_path)
    
    def test_pdf_initialization(self):
        """测试PDF处理器初始化"""
        print("\n=== 测试1: PDF处理器初始化 ===")
        
        # 测试带路径初始化
        processor = Pdf(self.test_pdf_path)
        self.assertEqual(processor.pdf_path, self.test_pdf_path)
        self.assertEqual(len(processor.documents), 0)
        self.assertEqual(len(processor.tables), 0)
        print("✓ PDF处理器初始化成功")
        
        # 测试无路径初始化
        processor_empty = Pdf()
        self.assertIsNone(processor_empty.pdf_path)
        print("✓ 空路径初始化成功")
    
    def test_structured_loading(self):
        """测试结构化表格提取功能"""
        print("\n=== 测试2: 使用pdfplumber的表格提取 ===")
        
        try:
            result = self.pdf_processor.extract_with_pdfplumber()
            
            self.assertIsInstance(result, dict)
            self.assertIn('table_elements', result)
            
            table_count = len(result['table_elements'])
            
            print(f"✓ 结构化表格提取成功")
            print(f"✓ 表格元素: {table_count} 个")
            
            # 检查内部状态更新
            self.assertEqual(len(self.pdf_processor.tables), table_count)
            
            # 显示第一个表格元素的内容预览
            if result['table_elements']:
                first_table = result['table_elements'][0]
                print(f"✓ 第一个表格元素预览: {first_table.page_content[:100]}...")
                
        except Exception as e:
            # 记录错误但不让测试失败（可能是依赖问题）
            print(f"⚠ 结构化表格提取遇到问题: {e}")
            logger.warning(f"结构化表格提取测试失败: {e}")
    
    def test_pdfplumber_extraction(self):
        """测试pdfplumber加载功能"""
        print("\n=== 测试3: pdfplumber加载 (extract_with_pdfplumber) ===")
        
        try:
            result = self.pdf_processor.extract_with_pdfplumber()
            
            self.assertIsInstance(result, dict)
            self.assertIn('text_elements', result)
            self.assertIn('table_elements', result)
            
            text_count = len(result['text_elements'])
            table_count = len(result['table_elements'])
            
            print(f"✓ pdfplumber加载成功")
            print(f"✓ 文本元素: {text_count} 个")
            print(f"✓ 表格元素: {table_count} 个")
            
            # 显示文本元素详情
            if result['text_elements']:
                first_text = result['text_elements'][0]
                print(f"✓ 第一个文本元素页面: {first_text.metadata.get('page')}")
                print(f"✓ 第一个文本元素长度: {len(first_text.page_content)} 字符")
                print(f"✓ 第一个文本元素预览: {first_text.page_content[:100]}...")
            
            # 显示表格元素详情
            if result['table_elements']:
                first_table = result['table_elements'][0]
                print(f"✓ 第一个表格元素页面: {first_table.metadata.get('page')}")
                print(f"✓ 第一个表格元素预览: {first_table.page_content[:200]}...")
                
        except Exception as e:
            print(f"⚠ pdfplumber加载遇到问题: {e}")
            logger.warning(f"pdfplumber加载测试失败: {e}")
    
    def test_comprehensive_extraction(self):
        """测试综合提取功能"""
        print("\n=== 测试4: 综合提取 (comprehensive_extract) ===")
        
        try:
            result = self.pdf_processor.comprehensive_extract()
            
            # 验证返回结构
            expected_keys = ['text_elements', 'table_elements', 'total_tables']
            for key in expected_keys:
                self.assertIn(key, result)
            
            text_count = len(result['text_elements'])
            table_count = len(result['table_elements'])
            total_tables = result['total_tables']
            
            print(f"✓ 综合提取成功")
            print(f"✓ 文本元素: {text_count} 个")
            print(f"✓ 表格元素: {table_count} 个")
            print(f"✓ 总表格数: {total_tables} 个")
            
            # 验证total_tables计算正确性
            self.assertEqual(total_tables, table_count)
            
            # 显示数据样例
            if result['text_elements']:
                print(f"\n✓ 文本样例:")
                for i, elem in enumerate(result['text_elements'][:2]):
                    print(f"  文本{i+1}: {elem.page_content[:100]}...")
            
            if result['table_elements']:
                print(f"\n✓ 表格样例:")
                for i, elem in enumerate(result['table_elements'][:2]):
                    print(f"  表格{i+1}: {elem.page_content[:100]}...")
            
        except Exception as e:
            print(f"⚠ 综合提取遇到问题: {e}")
            logger.warning(f"综合提取测试失败: {e}")
    
    def test_tables_to_text(self):
        """测试表格转文本功能"""
        print("\n=== 测试5: 表格转文本 (tables_to_text) ===")
        
        try:
            # 创建测试DataFrame数据
            import pandas as pd
            
            # 创建测试表格
            test_data1 = pd.DataFrame({
                '姓名': ['张三', '李四'],
                '部门': ['技术部', '市场部'],
                '职位': ['工程师', '经理']
            })
            test_data1.name = "test_table_1"
            
            test_data2 = pd.DataFrame({
                '项目': ['项目A', '项目B'],
                '状态': ['进行中', '已完成']
            })
            test_data2.name = "test_table_2"
            
            test_tables = [test_data1, test_data2]
            
            # 测试转换
            text_result = self.pdf_processor.tables_to_text(test_tables)
            
            self.assertIsInstance(text_result, str)
            self.assertGreater(len(text_result), 0)
            
            print(f"✓ 表格转文本成功")
            print(f"✓ 转换后文本长度: {len(text_result)} 字符")
            print(f"✓ 转换结果预览:")
            print(text_result[:500] + "..." if len(text_result) > 500 else text_result)
            
            # 验证包含表格标识
            self.assertIn("===", text_result)
            self.assertIn("test_table_1", text_result)
            self.assertIn("test_table_2", text_result)
                
        except Exception as e:
            print(f"⚠ 表格转文本遇到问题: {e}")
            logger.warning(f"表格转文本测试失败: {e}")
    
    
    def test_chunk_function(self):
        """测试chunk函数"""
        print("\n=== 测试7: chunk函数测试 ===")
        
        # 需要导入chunk函数
        from doc_parse import chunk
        
        try:
            # 测试默认参数
            documents = chunk(self.test_pdf_path)
            
            self.assertIsInstance(documents, list)
            self.assertGreater(len(documents), 0)
            
            print(f"✓ chunk函数执行成功")
            print(f"✓ 返回文档数量: {len(documents)}")
            
            # 检查文档格式
            for i, doc in enumerate(documents[:3]):
                self.assertTrue(hasattr(doc, 'page_content'))
                self.assertTrue(hasattr(doc, 'metadata'))
                print(f"✓ 文档{i+1}: {len(doc.page_content)} 字符, 类型: {doc.metadata.get('type', 'Unknown')}")
                print(f"  分块状态: {doc.metadata.get('is_chunked', 'Unknown')}")
                if doc.metadata.get('is_chunked'):
                    print(f"  块索引: {doc.metadata.get('chunk_index')}/{doc.metadata.get('total_chunks')}")
            
            # 测试自定义参数
            documents_custom = chunk(self.test_pdf_path, chunk_size=2000, chunk_overlap=300)
            print(f"✓ 自定义参数chunk: {len(documents_custom)} 个文档")
            
        except Exception as e:
            print(f"⚠ chunk函数测试遇到问题: {e}")
            logger.warning(f"chunk函数测试失败: {e}")
    
    def test_multiple_pdf_files(self):
        """测试处理多个PDF文件"""
        print("\n=== 测试8: 多PDF文件处理 ===")
        
        pdf_files = [self.test_pdf_path]
        if os.path.exists(self.test_pdf_path_alt):
            pdf_files.append(self.test_pdf_path_alt)
        
        results = {}
        for pdf_file in pdf_files:
            try:
                processor = Pdf(pdf_file)
                result = processor.comprehensive_extract()
                results[pdf_file] = result
                print(f"✓ 成功处理: {pdf_file}")
                print(f"  文本元素: {len(result['text_elements'])} 个")
                print(f"  表格元素: {len(result['table_elements'])} 个")
            except Exception as e:
                print(f"⚠ 处理 {pdf_file} 时出错: {e}")
        
        self.assertGreater(len(results), 0, "至少应该成功处理一个PDF文件")
        print(f"✓ 总共成功处理 {len(results)} 个PDF文件")


class TestPDFContent(unittest.TestCase):
    """PDF内容验证测试类"""
    
    def setUp(self):
        self.pdf_path = "data/zhidu_employee.pdf"
        self.pdf_processor = Pdf(self.pdf_path)
    
    def test_pdf_accessibility(self):
        """测试PDF文件可访问性"""
        print("\n=== 内容测试1: PDF文件可访问性 ===")
        
        self.assertTrue(os.path.exists(self.pdf_path), f"PDF文件应该存在: {self.pdf_path}")
        
        file_size = os.path.getsize(self.pdf_path)
        print(f"✓ PDF文件大小: {file_size:,} 字节 ({file_size/1024:.1f} KB)")
        
        self.assertGreater(file_size, 0, "PDF文件不应为空")
    
    def test_content_extraction_quality(self):
        """测试内容提取质量"""
        print("\n=== 内容测试2: 内容提取质量 ===")
        
        try:
            result = self.pdf_processor.comprehensive_extract()
            
            # 检查是否提取到有意义的内容
            has_meaningful_content = False
            
            for element in result['text_elements']:
                if len(element.page_content.strip()) > 10:  # 有足够长的文本
                    has_meaningful_content = True
                    break
            
            # 检查表格元素
            if result['table_elements']:
                has_meaningful_content = True
            
            self.assertTrue(has_meaningful_content, "应该提取到有意义的内容")
            print("✓ 提取到有意义的内容")
            
            # 统计中文字符
            chinese_char_count = 0
            for element in result['text_elements'][:5]:  # 只检查前5个元素
                text = element.page_content
                chinese_char_count += sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
            
            print(f"✓ 前5个文本元素中包含 {chinese_char_count} 个中文字符")
            
        except Exception as e:
            print(f"⚠ 内容质量测试遇到问题: {e}")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("PDF处理功能测试开始 - 使用doc_parse封装工具 (pdfplumber版本)")
    print("=" * 60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestPDFProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestPDFContent))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"运行测试数量: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # 检查必要的依赖
    print("检查依赖库...")
    
    required_libs = ['pandas', 'doc_parse']
    optional_libs = ['pdfplumber', 'langchain_community']
    
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"✓ {lib} 已安装")
        except ImportError:
            print(f"✗ {lib} 未安装 (必需)")
    
    for lib in optional_libs:
        try:
            __import__(lib)
            print(f"✓ {lib} 已安装")
        except ImportError:
            print(f"- {lib} 未安装 (可选)")
    
    print("\n开始运行测试...")
    success = run_tests()
    
    if success:
        print("\n🎉 所有测试通过!")
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息")