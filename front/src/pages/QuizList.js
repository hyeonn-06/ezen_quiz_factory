import axios from "axios";
import { useEffect, useState } from "react";
import { API_IP } from "../Config";
import { useNavigate, useSearchParams } from "react-router-dom";
import dayjs from "dayjs";

const QuizList = () => {
    const [list, setList] = useState([]);
    const [pageObject, setPageObject] = useState(null);

    const [searchParams, setSearchParams] = useSearchParams();
    const currentPage = Number.parseInt(searchParams.get("page") || "1")

    const navigate = useNavigate();

    useEffect(() => {
        const fetchData = async (page = 1) => {
            const requestUrl = `http://${API_IP}/member/quiz/list.do?page=${page}&perPageNum=10`
            try {
                const res = await axios.get(requestUrl, {
                    withCredentials: true
                });
                if (res.data && res.data.list) {
                    setList(res.data.list);
                    setPageObject(res.data.pageObject);
                } else {
                    setList([]);
                }

            } catch (err) {
                if (err.response) {
                    if (err.response.status === 403) {
                        alert("로그인 후 이용 가능합니다.");
                        navigate("/auth/login");
                    }
                    else {
                        alert("예상치 못한 오류가 발생했습니다. 잠시후 다시 시도해주시기 바랍니다.");
                        navigate("/");
                    }
                }
            }
        }
        fetchData(currentPage);
    }, [currentPage]);

    const submit = (quiz_no) => {
        navigate(`/member/quiz/view?quiz_no=${quiz_no}&page=${currentPage}`)
    }

    const handleCreateBtnClick = () => {
        navigate("/member/quiz/create");
    }

    const handlePageClick = (page) => {
        setSearchParams({ page: page.toString() })
    }

    const handlePrevGroup = () => {
        if (pageObject && pageObject.startPage > 1) {
            const prevPage = pageObject.startPage - 1
            setSearchParams({ page: prevPage.toString() })
        }
    }

    const handleNextGroup = () => {
        if (pageObject && pageObject.endPage < pageObject.totalPage) {
            const nextPage = pageObject.endPage + 1
            setSearchParams({ page: nextPage.toString() })
        }
    }

    const renderPagination = () => {
        if (!pageObject) return null;
        const pages = [];
        if (pageObject.startPage > 1) {
            pages.push(
                <li key="prev-group" className="page-item">
                    <button className="page-link" aria-label="Previous group"
                        onClick={handlePrevGroup}>
                        «
                    </button>
                </li>
            );
        }

        if (pageObject.page > 1) {
            pages.push(
                <li key="prev" className="page-item">
                    <button className="page-link"
                        aria-label="Previous"
                        onClick={() => handlePageClick(pageObject.page - 1)}>
                        ‹
                    </button>
                </li>
            );
        }

        for (let i = pageObject.startPage; i <= pageObject.endPage; i++) {
            pages.push(
                <li key={i} className={`page-item ${currentPage === i ? 'active' : ''}`}>
                    <button className="page-link" onClick={() => handlePageClick(i)} >
                        {i}
                    </button>
                </li>
            );
        }

        if (pageObject.page < pageObject.totalPage) {
            pages.push(
                <li key="next" className="page-item">
                    <button className="page-link" aria-label=" Next" onClick={() => handlePageClick(pageObject.page + 1)}>
                        ›
                    </button>
                </li>
            );
        }

        if (pageObject.endPage < pageObject.totalPage) {
            pages.push(
                <li key="next-group" className="page-item">
                    <button className="page-link" aria-label="Next group"
                        onClick={handleNextGroup}>
                        »
                    </button>
                </li>
            );
        }
        return pages;
    }
    return (
        <div className="container mt-5">
            <div className="d-flex justify-content-between align-items-center">
                <h1 className="mb-2">퀴즈 리스트</h1>
                <button className="btn btn-primary" onClick={handleCreateBtnClick}>생성</button>
            </div>
            <table className="table table-hover">
                <thead>
                    <tr>
                        <th className="col-1 text-center">No</th>
                        <th className="col-6">제목</th>
                        <th className="col-3">ID</th>
                        <th>생성일자</th>
                    </tr>
                </thead>
                <tbody>
                    {list.map(vo => (
                        <tr key={vo.quiz_no} onClick={() => submit(vo.quiz_no)}>
                            <td className="col-1 text-center">{vo.quiz_no}</td>
                            <td className="col-6">{vo.title}</td>
                            <td className="col-3">{vo.member_id}</td>
                            <td>{dayjs(vo.created_at).format('YYYY-MM-DD')}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
            <div className="d-flex justify-content-center mt-3">
                <ul className="pagination">
                    {renderPagination()}
                </ul>
            </div>
        </div>
    )
}
export default QuizList;